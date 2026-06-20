# Generative Video for Humanoid Control

Grounding generated robot videos into explicit geometry and motion for control.

Given a natural-language instruction and a single RGB image of a [Unitree G1](https://www.unitree.com/g1)
in a static scene, this project generates a short video of the robot performing the
instruction and grounds that video into a **metric, gravity-aligned scene mesh** (and,
in progress, a **parameterized reference trajectory**) for downstream simulation and
tracking-policy training.

The idea: instead of mapping language directly to actions, let a video model *imagine*
the task in the target scene, then recover the geometry and motion needed for control
from the generated pixels. See the accompanying paper, *Generative Video for Humanoid
Control*, for the full motivation and method.

## Pipeline

```
instruction + image
        │
        ▼
┌──────────────┐     ┌─────────────────────────────────────────────────────────┐
│  video-gen   │     │                    video-processing                     │
│  (Veo 3.1)   │ ──▶ │  sam3 ─▶ moge ─▶ geocrafter ─▶ sfm ─▶ alignment ─▶ mesh  │
└──────────────┘     └─────────────────────────────────────────────────────────┘
   generated video                        metric scene mesh (.stl)
```

| Stage         | Module          | What it does                                                          |
| ------------- | --------------- | -------------------------------------------------------------------- |
| Generation    | `video-gen`     | Veo 3.1 generates a short rollout from the instruction + seed image.  |
| Segmentation  | `sam3`          | SAM 3 masks the robot from a text prompt.                            |
| Geometry      | `moge`          | MoGe-2 predicts per-frame metric point maps.                        |
| Geometry      | `geocrafter`    | GeometryCrafter denoises the point maps into a temporally coherent sequence. |
| Registration  | `sfm`           | Robust 3D-3D alignment registers static scene geometry across frames. |
| Alignment     | `alignment`     | Manual keypoints fix a metric, gravity-aligned coordinate frame.     |
| Reconstruction| `mesh`          | Poisson surface reconstruction exports a simulation-ready mesh.      |
| *(in progress)* | robot pose    | Estimate G1 root pose + joints into a reference trajectory.          |

## Repository layout

- **[`video-gen/`](video-gen/)** — video and image generation via the Gemini / Veo API.
- **[`video-processing/`](video-processing/)** — the scene-reconstruction pipeline.

Each section has its own README with setup and usage details.

## Scope

Static scenes; humanoid locomotion and navigation; generated videos depict the target
robot embodiment. No manipulation or dynamic object interaction. This is an early
research prototype: the scene side of the pipeline is implemented; robot pose
estimation, policy training, and physical deployment are in progress.

## Credits

This project stands on several open models and methods:

- [Veo 3.1](https://deepmind.google/models/veo/) (Google DeepMind) — image-conditioned video generation.
- [SAM 3](https://github.com/facebookresearch/sam3) (Meta) — promptable concept segmentation.
- [MoGe / MoGe-2](https://github.com/microsoft/MoGe) (Microsoft) — monocular metric geometry.
- [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter) (Tencent ARC, ICCV 2025) — temporally consistent video point maps.
- [Open3D](https://www.open3d.org/) — Poisson surface reconstruction and point-cloud tooling.

The overall framing — reconstructing scene + motion from video to train a contextual
humanoid policy — follows [VideoMimic](https://github.com/hongsukchoi/VideoMimic)
(Choi et al., CoRL 2025), differing mainly in using *generated* rather than captured video.
