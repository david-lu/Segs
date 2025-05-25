#!/bin/bash
set -euo pipefail

start=$(date +%s)


# === Colors ===
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# === Function: Print error and exit ===
die() {
  echo -e "${RED}Error:${NC} $1" >&2
  exit 1
}

# === Parse arguments ===
input_video="${1:-}"
[ -n "$input_video" ] || die "Missing input video. Usage: $0 <input_video>"
[ -f "$input_video" ] || die "File '$input_video' does not exist."

# === Check dependencies ===
command -v python >/dev/null 2>&1 || die "'python' not found in PATH."
[ -f core/utils/run_inference.py ] || die "Missing: core/utils/run_inference.py"

# === Run inference ===
echo -e "${YELLOW}Running inference on '$input_video'...${NC}"

python core/utils/run_inference.py \
  --video_path "$input_video" \
  --gpus 0 1 2 3 \
  --depths \
  --tracks \
  --dinos \
  --e || die "Depth/track/dino inference failed."

python core/utils/run_inference.py \
  --video_path "$input_video" \
  --motin_seg_dir 'data/moseg_output' \
  --config_file 'configs/example_train.yaml' \
  --gpus 0 1 2 3 \
  --motion_seg_infer \
  --e || die "Motion segmentation inference failed." \
  --step 2 \
  --grid_size 64

python core/utils/run_inference.py \
  --video_path "$input_video" \
  --sam2dir 'data/moset_results' \
  --motin_seg_dir 'data/moseg_output' \
  --gpus 0 1 2 3 \
  --sam2 \
  --e || die "SAM2 inference failed."

echo -e "${YELLOW}✅ All inference steps completed successfully.${NC}"

end=$(date +%s)
echo "Elapsed time: $((end - start)) seconds"