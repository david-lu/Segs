#!/bin/sh
set -e

# === Install PyTorch 2.4.1 stack (CUDA 12.1) ===
pip install --no-deps torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# === Install xformers compatible with torch 2.4.1 ===
pip install xformers==0.0.28.post1

# === Rebuild and install sam2 with native extensions ===
sh build_sam.sh

(
  cd sam2/checkpoints || exit 1
  ./download_ckpts.sh
)

# === Install tapnet ===
(
  cd preproc/tapnet || exit 1
  pip install --no-deps .
)

# === Install all remaining dependencies (without letting them modify torch) ===
pip install --no-deps -r requirements.txt

# === Download checkpoints ===
wget -O models/moseg.pth https://huggingface.co/Changearthmore/moseg/resolve/main/moseg.pth
mkdir -p preproc/tapnet/checkpoints
wget -O preproc/tapnet/checkpoints/bootstapir_checkpoint_v2.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
