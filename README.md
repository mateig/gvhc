# Generative Video for Humanoid Control

Work in progress.

The idea here: give it a natural-language instruction and a single image of a Unitree G1
robot in a scene, and it generates a short video of the robot doing the task, then
reconstructs a metric, gravity-aligned 3D mesh of the scene out of that video. The point
is to recover enough geometry and motion to train a humanoid control policy in simulation.

There's a paper, *Generative Video for Humanoid Control*, with the full motivation and method.

## Layout

- `video-gen/` — generates the video and seed images (Google Gemini / Veo 3.1)
- `video-processing/` — reconstructs the scene mesh from that video

Each folder has its own README.

## Status

The scene-reconstruction side works. Robot pose estimation, policy training, and
real-world deployment aren't done yet.

## Credits

Built on Veo 3.1 (Google DeepMind), [SAM 3](https://github.com/facebookresearch/sam3),
[MoGe-2](https://github.com/microsoft/MoGe),
[GeometryCrafter](https://github.com/TencentARC/GeometryCrafter), and
[Open3D](https://www.open3d.org/). The overall approach follows
[VideoMimic](https://github.com/hongsukchoi/VideoMimic), just with generated video instead
of real captured video.
