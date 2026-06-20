# video-gen

This generates the robot video (and seed images) with Google's Gemini / Veo 3.1 API. The
video it produces is what `../video-processing` takes as input.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...
```

You'll need a Gemini API key in `GEMINI_API_KEY`.

## Running it

To generate a video, edit the prompt and the input/output paths near the top of
`run_video_gen.py`, then:

```bash
.venv/bin/python run_video_gen.py
```

To generate or edit a seed image:

```bash
.venv/bin/python run_image_gen.py input.jpg output.jpg "your prompt"
```

`run_pcd2mesh.py` is a separate point-cloud-to-mesh experiment I left in here for
reference — it's not part of the main flow.
