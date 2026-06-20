"""Convert the source mp4 into a uint8 video npz."""

import os

import cv2
import numpy as np

from scripts.config import cfg


def main() -> None:
    data = cfg["data"] + "/"
    video_source = data + cfg["video"]["source"]
    video_npz = data + cfg["video"]["npz"]

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_source}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames read from {video_source}")

    video = np.stack(frames).astype(np.uint8)
    os.makedirs(os.path.dirname(video_npz), exist_ok=True)
    np.savez(video_npz, video=video)


if __name__ == "__main__":
    main()
