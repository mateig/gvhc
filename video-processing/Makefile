# ssh root@64.247.201.32 -p 11574 -i ~/.ssh/id_ed25519
REMOTE = root@64.247.201.32
PORT = 11574
KEY = ~/.ssh/id_ed25519
SSH = ssh $(REMOTE) -p $(PORT) -i $(KEY)
RSYNC = rsync -az -e "ssh -p $(PORT) -i $(KEY)" --exclude '__pycache__' --exclude '*.pyc'
DIR = /root/working/stage-1
PY = .venv/bin/python -u -m
SRC = src scripts requirements.txt pyproject.toml config.yaml

.PHONY: render

install:
	python -m venv .venv && .venv/bin/pip install -r requirements-local.txt

fmt:
	.venv/bin/ruff format --no-cache . && .venv/bin/ruff check --no-cache --fix .

convert:
	$(PY) scripts.convert

pick:
	$(PY) scripts.pick

deploy:
	$(SSH) "apt update && apt install -y rsync"
	$(SSH) "mkdir -p $(DIR)"
	$(RSYNC) $(SRC) $(REMOTE):$(DIR)/

deploy-venv:
	$(SSH) "cd $(DIR) && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
	$(SSH) -t "cd $(DIR) && .venv/bin/hf auth login"

deploy-sam3:
	$(SSH) "cd $(DIR) && python -m venv .venv-sam3 && .venv-sam3/bin/pip install torch torchvision einops opencv-python-headless pycocotools psutil pyyaml && .venv-sam3/bin/pip install git+https://github.com/facebookresearch/sam3.git"

download-weights:
	$(SSH) "cd $(DIR) && .venv/bin/hf download Ruicheng/moge-2-vitl model.pt --local-dir weights/moge"
	$(SSH) "cd $(DIR) && .venv/bin/hf download stabilityai/stable-video-diffusion-img2vid-xt --include 'image_encoder/*' --include 'vae/*' --include 'scheduler/*' --exclude '*.fp16.*' --local-dir weights/geocrafter"
	$(SSH) "cd $(DIR) && .venv/bin/hf download TencentARC/GeometryCrafter --include 'unet_diff/*' --include 'point_map_vae/*' --local-dir weights/geocrafter"

upload:
	$(SSH) "mkdir -p $(DIR)/data"
	$(RSYNC) data/video.npz data/alignment.npz data/keypoints.npz $(REMOTE):$(DIR)/data/

run-sam3:
	$(SSH) "cd $(DIR) && .venv-sam3/bin/python -u -m scripts.run_sam3"

run:
	$(SSH) "cd $(DIR) && .venv/bin/python -u -m scripts.run"

render:
	$(PY) scripts.render
