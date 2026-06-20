# video-gen

Generates the robot video (and seed images) using Google's Gemini / Veo 3.1 API. The
output video is the input to `../video-processing`.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...
```

A Gemini API key must be set in `GEMINI_API_KEY`.

## Usage

To generate a video, set the prompt and input/output paths near the top of
`run_video_gen.py`, then run:

```bash
.venv/bin/python run_video_gen.py
```

To generate or edit a seed image:

```bash
.venv/bin/python run_image_gen.py input.jpg output.jpg "your prompt"
```

`run_pcd2mesh.py` is a standalone point-cloud-to-mesh experiment, kept for reference. It is
not part of the main pipeline.
