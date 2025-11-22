import torch
from torch import nn
from transformers import StoppingCriteria, GenerationConfig, StoppingCriteriaList, LogitsProcessorList, GenerationConfig, LogitsProcessorList


class EosStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1].eq(self.eos_token_id).any().item()
    

class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # input_ids: [batch, seq_len]
        return input_ids.shape[1] >= self.max_length


def generate_cd(clean_inputs, corrupt_inputs, model, processor, cd_alpha, cd_beta, use_cd):
    with torch.inference_mode():
        generation_config, clean_model_kwargs = model._prepare_generation_config(
            GenerationConfig(return_dict_in_generate=True, output_scores=True, do_sample=False, num_beams=1,
                             pad_token_id=processor.tokenizer.pad_token_id), None, **clean_inputs
        )
        _, corrupt_model_kwargs = model._prepare_generation_config(
            GenerationConfig(return_dict_in_generate=True, output_scores=True, do_sample=False, num_beams=1,
                             pad_token_id=processor.tokenizer.pad_token_id), None, **corrupt_inputs
        )
        eos_token_id = processor.tokenizer.eos_token_id
        stopping_criteria = StoppingCriteriaList([
            EosStoppingCriteria(eos_token_id),
            MaxLengthCriteria(8192)
        ])
        pad_token_id = processor.tokenizer.pad_token_id
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        logits_processor = LogitsProcessorList()
        kwargs_has_attention_mask = clean_model_kwargs.get("attention_mask", None) is not None
        clean_inputs_tensor, model_input_name, clean_model_kwargs = model._prepare_model_inputs(
            None, generation_config.bos_token_id, clean_model_kwargs
        )
        corrupt_inputs_tensor, model_input_name, corrupt_model_kwargs = model._prepare_model_inputs(
            None, generation_config.bos_token_id, corrupt_model_kwargs
        )
        device = clean_inputs_tensor.device
        model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        input_ids = clean_inputs_tensor if model_input_name == "input_ids" else clean_model_kwargs.pop("input_ids")
        corrupt_input_ids = corrupt_inputs_tensor if model_input_name == "input_ids" else corrupt_model_kwargs.pop("input_ids")

        if corrupt_input_ids.shape[1] < input_ids.shape[1]:
            pad_length = input_ids.shape[1] - corrupt_input_ids.shape[1]
            padding = torch.full(
                (corrupt_input_ids.shape[0], pad_length),
                processor.tokenizer.pad_token_id,
                dtype=corrupt_input_ids.dtype,
                device=corrupt_input_ids.device
            )
            corrupt_input_ids = torch.cat([corrupt_input_ids, padding], dim=1)
            
            if "attention_mask" in corrupt_model_kwargs:
                mask_padding = torch.zeros(
                    (corrupt_model_kwargs["attention_mask"].shape[0], pad_length),
                    dtype=corrupt_model_kwargs["attention_mask"].dtype,
                    device=corrupt_model_kwargs["attention_mask"].device
                )
                corrupt_model_kwargs["attention_mask"] = torch.cat(
                    [corrupt_model_kwargs["attention_mask"], mask_padding], dim=1
                )

        clean_model_kwargs["logits_to_keep"] = 1
        clean_model_kwargs["use_cache"] = True
        corrupt_model_kwargs["logits_to_keep"] = 1
        corrupt_model_kwargs["use_cache"] = True
        batch_size, cur_len = input_ids.shape[:2]
        clean_model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, clean_model_kwargs)
        corrupt_model_kwargs = model._get_initial_cache_position(cur_len, corrupt_input_ids.device, corrupt_model_kwargs)
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        while model._has_unfinished_sequences(this_peer_finished, False, device=input_ids.device):
            clean_model_inputs = model.prepare_inputs_for_generation(input_ids, **clean_model_kwargs)
            clean_outputs = model(**clean_model_inputs, return_dict=True)
            clean_model_kwargs = model._update_model_kwargs_for_generation(
                clean_outputs,
                clean_model_kwargs,
                is_encoder_decoder=False,
            )
            next_clean_logits = clean_outputs.logits[:, -1, :].to(dtype=torch.bfloat16, device=input_ids.device)
            corrupt_model_inputs = model.prepare_inputs_for_generation(corrupt_input_ids, **corrupt_model_kwargs)
            corrupt_outputs = model(**corrupt_model_inputs, return_dict=True)
            corrupt_model_kwargs = model._update_model_kwargs_for_generation(
                corrupt_outputs,
                corrupt_model_kwargs,
                is_encoder_decoder=False,
            )
            next_corrupt_logits = corrupt_outputs.logits[:, -1, :].to(dtype=torch.bfloat16, device=input_ids.device)
            cutoff = torch.log(torch.tensor(cd_beta, device=next_clean_logits.device)) + \
                    next_clean_logits.max(dim=-1, keepdim=True).values
            diffs = (1 + cd_alpha) * next_clean_logits - cd_alpha * next_corrupt_logits
            cd_logits = diffs.masked_fill(next_clean_logits < cutoff, -float("inf"))
            cd_logits = logits_processor(input_ids, cd_logits)
            next_token_scores = cd_logits
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
            del clean_outputs, corrupt_outputs
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, next_token_scores)
            this_peer_finished = unfinished_sequences.max() == 0
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            corrupt_input_ids = torch.cat([corrupt_input_ids, next_tokens[:, None]], dim=-1)
        return input_ids