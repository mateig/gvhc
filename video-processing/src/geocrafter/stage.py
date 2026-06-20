"""Stage wrapper: loads GeoCrafter weights and runs on video + MoGe points."""

import os

import numpy as np
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection

from src.geocrafter.geometry import produce_priors
from src.geocrafter.pipeline import GeoCrafterPipeline, reconstruct_points
from src.geocrafter.unet import UNet
from src.geocrafter.vae import PMapVAE


def load_pipeline(weights: str, device: torch.device) -> GeoCrafterPipeline:
    root = weights
    image_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(root, "image_encoder"), torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(os.path.join(root, "vae")).to(device).eval()
    unet = UNet.from_pretrained(os.path.join(root, "unet_diff")).to(device).eval()
    pmap_vae = PMapVAE.from_pretrained(os.path.join(root, "point_map_vae")).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(root, "scheduler"))
    return GeoCrafterPipeline(image_encoder, vae, unet, pmap_vae, scheduler)


def run(
    video: str,
    moge: str,
    output: str,
    weights: str,
    process_resolution: int = 1024,
    steps: int = 25,
    window_size: int = 144,
    overlap: int = 25,
    batch_size: int = 8,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = load_pipeline(weights, device)

    # points: (T, H, W, 3) float32 camera-frame, NaN where invalid
    moge_data = np.load(moge)
    points = torch.tensor(moge_data["points"], device=device)

    disparity, point_map, intrinsic_map, valid_mask, depth_norm = produce_priors(points)
    del points

    # (T, H, W, 3) uint8 RGB
    video_np = np.load(video)["video"]
    frames = torch.tensor(video_np.astype(np.float32) / 255.0, device=device).permute(0, 3, 1, 2)
    H, W = frames.shape[-2:]
    del video_np

    with torch.inference_mode():
        intr, log_depth, vmask = pipeline(
            frames,
            disparity,
            point_map,
            intrinsic_map,
            valid_mask,
            process_resolution=process_resolution,
            num_inference_steps=steps,
            window_size=window_size,
            overlap=overlap,
            batch_size=batch_size,
        )
        points = reconstruct_points(intr, log_depth, vmask, depth_norm, (H, W))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    # points: (T, H, W, 3) float32 camera-frame, NaN where invalid
    np.savez(output, points=points.cpu().numpy())
