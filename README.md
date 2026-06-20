# Generative Video for Humanoid Control (GVHC)

Work in progress.

Given a natural-language instruction and a single image of a Unitree G1 robot in a scene,
the system generates a short video of the robot performing the task and reconstructs a
metric, gravity-aligned 3D mesh of the scene from that video. The goal is to recover
sufficient geometry and motion to train a humanoid control policy in simulation.

The paper, *Generative Video for Humanoid Control*, describes the full motivation and method.

## Layout

- `video-gen/` — generates the video and seed images (Google Gemini / Veo 3.1)
- `video-processing/` — reconstructs the scene mesh from that video

Each folder has its own README.

## Status

Scene reconstruction is implemented. Robot pose estimation, policy training, and
real-world deployment are not yet complete.

## Built on

Veo 3.1 (Google DeepMind), [SAM 3](https://github.com/facebookresearch/sam3),
[MoGe-2](https://github.com/microsoft/MoGe),
[GeometryCrafter](https://github.com/TencentARC/GeometryCrafter), and
[Open3D](https://www.open3d.org/).
