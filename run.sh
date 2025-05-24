#!/bin/bash
set -e

# === Function: Resize to max side ===
resize_to_max() {
  input="$1"
  output="$2"
  max_size="${3:-640}"

  if [ ! -f "$input" ]; then
    echo "Error: Input file '$input' does not exist."
    return 1
  fi

  ffmpeg -y -i "$input" \
    -vf "scale='if(gt(iw,ih),${max_size},-2)':'if(gt(ih,iw),${max_size},-2)'" \
    -c:a copy "$output"

  echo "$output"
}

# === Inputs ===
input_video="$1"              # e.g. data/cinderella_1038.mkv
resized_video="$2"            # e.g. data/cinderella_1038_resized.mp4

if [ -z "$input_video" ] || [ -z "$resized_video" ]; then
  echo "Usage: $0 <input_video> <resized_output_video>"
  exit 1
fi

# === Resize video ===
resize_to_max "$input_video" "$resized_video" 640

# === Run inference using resized video ===

python core/utils/run_inference.py \
  --video_path "$resized_video" \
  --gpus 0 1 2 3 \
  --depths \
  --tracks \
  --dinos \
  --e

python core/utils/run_inference.py \
  --video_path "$resized_video" \
  --motin_seg_dir 'data/moseg_output' \
  --config_file 'configs/example_train.yaml' \
  --gpus 0 1 2 3 \
  --motion_seg_infer \
  --e

python core/utils/run_inference.py \
  --video_path "$resized_video" \
  --sam2dir 'data/moset_results' \
  --motin_seg_dir 'data/moseg_output' \
  --gpus 0 1 2 3 \
  --sam2 \
  --e
