# video-processing

Reconstructs a metric, gravity-aligned 3D mesh of the scene from a generated robot video.

The pipeline runs as a sequence of stages. Each stage reads and writes `.npz` files in
`data/`, so stages are independent and can be re-run individually.

1. **sam3** — segment the robot from the video ([SAM 3](https://github.com/facebookresearch/sam3))
2. **moge** — per-frame metric point maps ([MoGe-2](https://github.com/microsoft/MoGe))
3. **geocrafter** — enforce temporal consistency on the point maps ([GeometryCrafter](https://github.com/TencentARC/GeometryCrafter))
4. **sfm** — register the static scene across frames (optical flow + RANSAC alignment)
5. **alignment** — fix a metric, gravity-aligned frame from a few picked keypoints
6. **mesh** — Poisson surface reconstruction into a `.stl` (Open3D)

Robot pose estimation is intended to follow this stage but is not yet implemented.

The GPU-heavy stages (sam3, moge, geocrafter) run on a remote machine. Video conversion,
keypoint picking, and rendering run locally. Paths and parameters are defined in
`config.yaml`.

## Setup

Local dependencies (Open3D and a few others):

```bash
make install
```

For the remote machine, set `REMOTE`, `PORT`, `KEY`, and `DIR` at the top of the
`Makefile`, then:

```bash
make deploy
make deploy-venv
make deploy-sam3
make download-weights
```

`deploy-sam3` builds SAM 3 in a separate venv because its dependencies conflict with the
other stages.

## Usage

Convert the video and pick keypoints locally, then upload to the remote:

```bash
make convert
make pick
make upload
```

Run the stages on the remote (sam3 is run separately due to the venv split):

```bash
make run-sam3
make run
```

Copy the resulting `.npz` and `.stl` files back into `data/`, then render the stages to
video for inspection:

```bash
make render
```

To re-run only part of the pipeline, use the `skip` lists in `config.yaml`; any stage
listed there is skipped.

All contents of `data/` are generated and git-ignored.
