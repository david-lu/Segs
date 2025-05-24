#!/bin/bash

# Deinitialize all submodules
git submodule deinit -f --all

# Remove all submodule metadata
rm -rf .git/modules/preproc/dinov2
rm -rf .git/modules/preproc/tapnet
rm -rf .git/modules/core/eval/davis2017-evaluation

# Optionally remove their working directories (if broken)
rm -rf preproc/dinov2
rm -rf preproc/tapnet
rm -rf core/eval/davis2017-evaluation

git submodule deinit -f --all
git submodule update --init --recursive --force

#!/bin/sh
set -e  # Exit on error

# Install sam2
(
  cd sam2 || exit
  pip install -e .

  cd checkpoints || exit
  ./download_ckpts.sh
)

# Install tapnet
(
  cd preproc/tapnet || exit
  pip install .
)

# Download TapNet checkpoint
mkdir -p preproc/tapnet/checkpoints
(
  cd preproc/tapnet/checkpoints || exit
  wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
)
