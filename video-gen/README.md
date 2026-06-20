# video-gen

Generate the robot video and seed images using Google's Gemini / Veo 3.1 API. The video
produced here is the input to `../video-processing`.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...   # your Gemini API key
```

## Usage

Generate a video (edit the prompt and paths at the top of the file first):

```bash
.venv/bin/python run_video_gen.py
```

Generate or edit a seed image:

```bash
.venv/bin/python run_image_gen.py <input.jpg> <output.jpg> "<prompt>"
```

`run_pcd2mesh.py` is a standalone point-cloud-to-mesh experiment, kept for reference.
