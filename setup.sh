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

# Download TapNet checkpoint
mkdir -p preproc/tapnet/checkpoints
(
  cd preproc/tapnet/checkpoints || exit 1
  wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
)
