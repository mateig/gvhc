"""Stage wrapper: loads MoGe weights and predicts per-frame points."""

import os

import numpy as np
import torch

from src.moge.geometry import recover_3d
from src.moge.model import MoGeModel


def load_model(weights: str, device: torch.device) -> MoGeModel:
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    model = MoGeModel(**ckpt["model_config"])
    state = dict(ckpt["model"])
    state.pop("encoder.backbone.mask_token", None)
    model.load_state_dict(state)
    return model.to(device).eval()


def run(
    video: str,
    output: str,
    weights: str,
    num_tokens: int = 4500,
    batch_size: int = 8,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)

    # (T, H, W, 3) uint8 RGB
    video_np = np.load(video)["video"]
    frames = torch.tensor(video_np.astype(np.float32) / 255.0, device=device).permute(0, 3, 1, 2)

    with torch.inference_mode():
        pts_chunks, scale_chunks, mask_chunks = [], [], []
        for i in range(0, len(frames), batch_size):
            pts, s, m = model(frames[i : i + batch_size], num_tokens)
            pts_chunks.append(pts)
            scale_chunks.append(s)
            mask_chunks.append(m)
        mask = torch.cat(mask_chunks) > 0.5
        points, _, _ = recover_3d(torch.cat(pts_chunks), torch.cat(scale_chunks), mask=mask)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    # points: (T, H, W, 3) float32 camera-frame, NaN where invalid
    np.savez(output, points=points.cpu().numpy())
