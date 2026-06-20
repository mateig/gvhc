# video-gen

Video and image generation via Google's [Gemini](https://ai.google.dev/) / [Veo 3.1](https://deepmind.google/models/veo/) API.

This is the first stage of the pipeline: it turns a natural-language instruction and a
single seed image into a short video of the robot carrying out the task in its scene.
That video is the input to [`../video-processing`](../video-processing).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...   # your Google AI Studio / Gemini API key
```

The Gemini client reads the key from the `GEMINI_API_KEY` environment variable.

## Usage

**Generate a video** (Veo 3.1, image-conditioned, 16:9 / 720p / 24 fps / 6 s by default):

```bash
.venv/bin/python run_video_gen.py
```

Edit the `prompt`, `seed_path`, and `save_path` at the top of `run_video_gen.py`.

**Edit / generate a seed image** (Gemini image model):

```bash
.venv/bin/python run_image_gen.py <input.jpg> <output.jpg> "<edit prompt>"
```

**Point cloud → mesh** (`run_pcd2mesh.py`) is a standalone experiment that scales a
captured point cloud to metric units using a known reference distance and reconstructs a
Poisson mesh of a masked region. It is superseded by the `mesh` stage in
`video-processing` but kept for reference.

## Files

| File                 | Purpose                                                        |
| -------------------- | ------------------------------------------------------------- |
| `lib/gemini.py`      | Thin wrappers around `generate_video` and `edit_image`.       |
| `lib/video.py`       | Load an mp4 into a `(T, H, W, 3)` numpy array.                |
| `run_video_gen.py`   | Image-conditioned video generation entry point.               |
| `run_image_gen.py`   | Image editing / generation entry point.                       |
| `run_pcd2mesh.py`    | Standalone point-cloud-to-mesh experiment.                    |
