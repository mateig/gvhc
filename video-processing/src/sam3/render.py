"""Render SAM3 masks overlaid on the source video."""

import os

import cv2
import numpy as np
from tqdm import tqdm


def render(
    video: str,
    masks: str,
    output: str,
    fps: int,
    alpha: float,
) -> None:
    frames = np.load(video)["video"]
    ms = np.load(masks)["masks"].any(axis=1)
    T, H, W, _ = frames.shape

    color = np.array([255, 0, 0], dtype=np.float32)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for t in tqdm(range(T)):
        frame = frames[t].astype(np.float32)
        m = ms[t]
        frame[m] = (1 - alpha) * frame[m] + alpha * color
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))

    writer.release()
