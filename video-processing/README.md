# video-processing

This takes a generated robot video and reconstructs a metric, gravity-aligned 3D mesh of
the scene from it.

It's split into stages that run one after another. Each stage reads and writes `.npz`
files in `data/`, so they're independent — you can re-run any one of them on its own.

The stages are:

1. **sam3** — segment the robot out of the video ([SAM 3](https://github.com/facebookresearch/sam3))
2. **moge** — per-frame metric point maps ([MoGe-2](https://github.com/microsoft/MoGe))
3. **geocrafter** — make those point maps temporally consistent ([GeometryCrafter](https://github.com/TencentARC/GeometryCrafter))
4. **sfm** — register the static scene across frames (optical flow + RANSAC alignment)
5. **alignment** — fix a metric, gravity-aligned frame from a few picked keypoints
6. **mesh** — Poisson surface reconstruction into a `.stl` (Open3D)

Robot pose estimation is meant to come after this but isn't done yet.

The GPU-heavy stages (sam3, moge, geocrafter) run on a remote machine. Converting the
video, picking keypoints, and rendering happen locally. Paths and parameters all live in
`config.yaml`.

## Setup

Local side — just need Open3D and a few things:

```bash
make install
```

For the remote box, edit `REMOTE`, `PORT`, `KEY`, and `DIR` at the top of the `Makefile`,
then:

```bash
make deploy
make deploy-venv
make deploy-sam3
make download-weights
```

`deploy-sam3` builds SAM 3 in its own venv because its dependencies don't play nice with
the rest.

## Running it

Convert the video and pick keypoints locally, then send everything to the remote:

```bash
make convert
make pick
make upload
```

Run the stages on the remote (sam3 is separate because of the venv split):

```bash
make run-sam3
make run
```

Pull the resulting `.npz` and `.stl` files back into `data/`, then render the stages to
video to check them:

```bash
make render
```

If you only want to re-run part of the pipeline, use the `skip` lists in `config.yaml` —
anything listed there gets skipped.

Everything under `data/` is generated, so it's git-ignored.
