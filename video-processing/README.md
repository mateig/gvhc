# video-processing

The scene-reconstruction pipeline. It takes a generated robot video and produces a
**metric, gravity-aligned static scene mesh** (`.stl`), saving a rendered video of every
intermediate stage for inspection.

```
video.npz ─▶ sam3 ─▶ moge ─▶ geocrafter ─▶ sfm ─▶ alignment ─▶ mesh.stl
```

The GPU stages (segmentation, geometry, diffusion) run on a remote machine; conversion,
keypoint picking, and Open3D rendering run locally. Each stage reads and writes plain
`.npz` artifacts under `data/`, so stages are independent and resumable.

## Stages

| Stage        | Source            | Output           | Based on                                                                 |
| ------------ | ----------------- | ---------------- | ------------------------------------------------------------------------ |
| `sam3`       | `src/sam3`        | `sam3.npz`       | [SAM 3](https://github.com/facebookresearch/sam3) — robot mask from the text prompt `"humanoid robot"`. |
| `moge`       | `src/moge`        | `moge.npz`       | [MoGe-2](https://github.com/microsoft/MoGe) (`Ruicheng/moge-2-vitl`) — per-frame metric point maps via a DINOv2 backbone. |
| `geocrafter` | `src/geocrafter`  | `geocrafter.npz` | [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter) — diffusion prior denoises MoGe point maps into a temporally coherent sequence. |
| `sfm`        | `src/sfm`         | `sfm.npz`        | Classical 3D-3D registration: DIS optical flow + RANSAC Kabsch + a multi-gap pose graph + Savitzky-Golay smoothing. Static-scene assumption removes the moving robot before matching. |
| `alignment`  | `src/alignment`   | `alignment.npz`  | Three manually picked scene keypoints + a known distance fix a metric, gravity-aligned frame. |
| `mesh`       | `src/mesh`        | `mesh.stl`       | [Open3D](https://www.open3d.org/) Poisson surface reconstruction of the static points around the robot. |

`src/render/pointcloud.py` renders the per-frame point clouds for each stage to mp4 via
Open3D. Robot pose estimation (`q_ref`) is the next stage and is not yet implemented.

## Setup

**Local** (conversion, keypoint picking, rendering — needs Open3D):

```bash
make install          # python -m venv .venv && pip install -r requirements-local.txt
```

**Remote GPU box** (the heavy stages). The `Makefile` rsyncs the code, builds two venvs
(SAM 3 is isolated in `.venv-sam3`), and downloads weights. Edit `REMOTE`, `PORT`, `KEY`,
and `DIR` at the top of the `Makefile` for your host, then:

```bash
make deploy           # rsync src/, scripts/, config, requirements to the remote
make deploy-venv      # build .venv + hf login
make deploy-sam3      # build the isolated SAM 3 venv
make download-weights # MoGe-2, SVD, and GeometryCrafter weights via huggingface-cli
```

## Workflow

```bash
# --- local ---
make convert          # data/video.mp4 -> data/video.npz
make pick             # click scene keypoints -> data/keypoints.npz
python -m scripts.camera   # drag to set a view, press S -> data/camera.json (for rendering)
make upload           # send video.npz, keypoints.npz, alignment.npz to the remote

# --- remote ---
make run-sam3         # SAM 3 segmentation (isolated venv)
make run              # moge -> geocrafter -> sfm -> alignment -> mesh

# --- local ---
# pull the resulting .npz / .stl back into data/, then:
make render           # render each stage's point cloud to mp4
```

## Configuration

All paths and per-stage parameters live in [`config.yaml`](config.yaml). Notable knobs:

- `skip.run` / `skip.render` — lists of stages to skip; the dispatchers
  ([`scripts/run.py`](scripts/run.py), [`scripts/render.py`](scripts/render.py)) run
  everything *not* listed, so you can re-run a single stage.
- `sam3.run.prompt` — the segmentation text prompt.
- `sfm.run` — flow gaps, correspondence sampling, RANSAC/cycle thresholds, pose-graph
  smoothing.
- `alignment.run` — keypoint indices (`p1`, `p2`, `p3`), `known_distance`, frame offset.
- `mesh.run` — crop `radius`, voxel size, outlier removal, Poisson `depth`, smoothing.

## Data artifacts

Everything under `data/` is generated (and git-ignored). Each `.npz` carries a small
header comment in the corresponding `stage.py` describing its arrays — e.g. `sfm.npz`
holds world-frame point maps `(T, H, W, 3)`, `alignment.npz` adds the rigid `extrinsic`
and metric `scale`, and `mesh.stl` is the final collision surface.
