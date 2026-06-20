# Generative Video for Humanoid Control

**Work in progress.**

Given a natural-language instruction and a single image of a Unitree G1 robot in a static
scene, this project generates a short video of the robot doing the task, then reconstructs
a metric, gravity-aligned 3D mesh of the scene from that video. The goal is to recover the
geometry and motion needed to train a humanoid control policy in simulation.

See the paper, *Generative Video for Humanoid Control*, for the full motivation and method.

## Layout

- `video-gen/` — generate the video and seed images (Google Gemini / Veo 3.1).
- `video-processing/` — reconstruct the scene mesh from the generated video.

Each folder has its own README.

## Status

The scene-reconstruction side is implemented. Robot pose estimation, policy training, and
real-world deployment are still in progress.

## Credits

Built on Veo 3.1 (Google DeepMind), [SAM 3](https://github.com/facebookresearch/sam3),
[MoGe-2](https://github.com/microsoft/MoGe),
[GeometryCrafter](https://github.com/TencentARC/GeometryCrafter), and
[Open3D](https://www.open3d.org/). The overall approach follows
[VideoMimic](https://github.com/hongsukchoi/VideoMimic), but uses generated video instead
of captured video.
