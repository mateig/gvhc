"""Interactive click-picker for keypoints on the first video frame."""

import os

import cv2
import numpy as np

from scripts.config import cfg


def collect_clicks(frame_bgr: np.ndarray) -> list[tuple[int, int]]:
    points_px: list[tuple[int, int]] = []
    display = frame_bgr.copy()

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points_px.append((x, y))
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("pick", display)

    cv2.imshow("pick", display)
    cv2.setMouseCallback("pick", on_click)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:
            break
        if key == 27:
            points_px.clear()
            break
    cv2.destroyAllWindows()
    return points_px


def main() -> None:
    data = cfg["data"] + "/"
    video_npz = data + cfg["video"]["npz"]
    keypoints_npz = data + cfg["keypoints"]["npz"]

    frame = cv2.cvtColor(np.load(video_npz)["video"][0], cv2.COLOR_RGB2BGR)
    clicks = collect_clicks(frame)
    if not clicks:
        return
    keypoints = np.array(clicks, dtype=np.int32)
    os.makedirs(os.path.dirname(keypoints_npz), exist_ok=True)
    np.savez(keypoints_npz, keypoints=keypoints)


if __name__ == "__main__":
    main()
