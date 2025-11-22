import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu
import torch
from transformers import CLIPModel, CLIPProcessor 

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

def load_video(video_path, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_segments, dtype=int)
    frames = [vr[i].asnumpy() for i in frame_indices]  

    return frames


def _find_scene_change_idx(frames_list, clip_model, clip_processor):
    device = clip_model.device
    inputs = clip_processor(
        images=frames_list,
        return_tensors="pt",
        padding=True,
        do_rescale=True
    ).to(device)

    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)

    embeddings = F.normalize(embeddings, p=2, dim=-1)
    sims = (embeddings[:-1] * embeddings[1:]).sum(dim=-1) 
    return sims.cpu().numpy()


def _apply_gaussian(img_uint8, mean=0.0, std=25.0):
    noise = np.random.normal(mean, std, img_uint8.shape)
    out = np.clip(img_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def add_pertubation(video_np, gaussian_std, THRESHOLD=0.8):
    sims = _find_scene_change_idx(video_np, clip_model, clip_processor)
    num_segments = len(sims)
    has_candidate = False
    targets = []
    min_idx = np.argmin(sims)
    if sims[min_idx] < THRESHOLD:
        has_candidate = True
        targets = list(range(min_idx + 1, num_segments)) 
    
    for i in targets:
        video_np[i] = _apply_gaussian(video_np[i], mean=0, std=gaussian_std)

    return video_np, has_candidate

