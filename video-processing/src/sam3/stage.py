"""Stage wrapper: SAM3 video mask propagation from a text prompt."""

import os
import tempfile

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor


def write_mp4(frames: np.ndarray, path: str, fps: int = 30) -> None:
    T, H, W = frames.shape[:3]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def propagate(predictor, video_path: str, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    resp = predictor.handle_request(request=dict(type="start_session", resource_path=video_path))
    session_id = resp["session_id"]
    predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
    )
    masks, obj_ids = [], None
    for resp in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        masks.append(resp["outputs"]["out_binary_masks"])
        obj_ids = resp["outputs"]["out_obj_ids"]
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return np.stack(masks), np.asarray(obj_ids)


def run(video: str, output: str, prompt: str) -> None:
    # (T, H, W, 3) uint8 RGB
    frames = np.load(video)["video"]

    predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        write_mp4(frames, tmp_path)
        masks, obj_ids = propagate(predictor, tmp_path, prompt)
    finally:
        os.unlink(tmp_path)
        predictor.shutdown()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    # masks: (T, N, H, W) bool; obj_ids: (N,) int64
    np.savez(output, masks=masks, obj_ids=obj_ids)
