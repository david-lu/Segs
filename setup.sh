#!/bin/sh
set -e  # Exit on error

# Install sam2
(
  cd sam2 || exit 1
  pip install -e .

  cd checkpoints || exit 1
  ./download_ckpts.sh
)

# Install tapnet
(
  cd preproc/tapnet || exit 1
  pip install .
)

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
wget -O models/moseg.pth https://huggingface.co/Changearthmore/moseg/resolve/main/moseg.pth
mkdir -p preproc/tapnet/checkpoints
wget -O preproc/tapnet/checkpoints/bootstapir_checkpoint_v2.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
