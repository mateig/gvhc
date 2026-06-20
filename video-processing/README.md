# video-processing

Reconstruct a metric, gravity-aligned 3D mesh of the scene from a generated robot video.

The pipeline runs as a sequence of stages, each reading and writing `.npz` files under
`data/`, so stages are independent and can be re-run on their own:

1. **sam3** — segment the robot ([SAM 3](https://github.com/facebookresearch/sam3)).
2. **moge** — per-frame metric point maps ([MoGe-2](https://github.com/microsoft/MoGe)).
3. **geocrafter** — make the point maps temporally consistent ([GeometryCrafter](https://github.com/TencentARC/GeometryCrafter)).
4. **sfm** — register the static scene across frames (optical flow + RANSAC alignment).
5. **alignment** — fix a metric, gravity-aligned frame from picked keypoints.
6. **mesh** — Poisson surface reconstruction into a `.stl` (Open3D).

Robot pose estimation is the next stage and is not implemented yet.

The GPU stages run on a remote machine; conversion, keypoint picking, and rendering run
locally. All paths and parameters live in `config.yaml`.

## Setup

Local (conversion, picking, rendering):

```bash
make install
```

Remote GPU box — edit `REMOTE`, `PORT`, `KEY`, `DIR` at the top of the `Makefile`, then:

```bash
make deploy           # copy code to the remote
make deploy-venv      # build the main venv
make deploy-sam3      # build the isolated SAM 3 venv
make download-weights # fetch model weights
```

## Usage

```bash
# local
make convert          # data/video.mp4 -> data/video.npz
make pick             # click scene keypoints -> data/keypoints.npz
make upload           # send inputs to the remote

# remote
make run-sam3         # segmentation
make run              # moge -> geocrafter -> sfm -> alignment -> mesh

# local (after pulling results back into data/)
make render           # render each stage to mp4
```

`config.yaml` has `skip` lists for the run and render steps — stages listed there are
skipped, so you can re-run just one stage at a time.

Everything under `data/` is generated and git-ignored.
