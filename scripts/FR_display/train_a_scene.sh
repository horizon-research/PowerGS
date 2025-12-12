#!/bin/bash

# Input scene name
scene_name=$1
q_required=$2

PSs=(3 7 12)
start_CKPTs=( 45000 50000 55000 )
save_CKPTs=( 50000 55000 60000 )
start_levels=( 1 2 3 )
save_levels=( 2 3 4 )

# Export paths
export m360_dataset_base_dir=/workspace/dataset/m360
export nerf_synthetic_dataset_base_dir=/workspace/dataset/nerf_synthetic
export code_dir=/workspace/PowerGS/
export method_name="FR_display_"$q_required""
export extend_dataset_base_dir="/workspace/dataset/extend"

# Scene type lists
mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
nerf_synthetic_scene=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
extend_scenes=("drjohnson" "playroom" "train" "truck")

# Function to check if a scene belongs to a list
function is_in_list() {
  local element=$1
  shift
  local list=("$@")
  for item in "${list[@]}"; do
    if [[ "$item" == "$element" ]]; then
      return 0
    fi
  done
  return 1
}

cd $code_dir

# Determine the scene type and run the appropriate command
if is_in_list "$scene_name" "${mipnerf360_outdoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 outdoor"
  python FR_Display/fr_display_L1.py -s "$m360_dataset_base_dir/$scene_name" -m "/workspace/PowerGS/models/$method_name/m360/$scene_name/levels/L1" -i images_4 --eval \
  --checkpoint_iterations 45000 --adjust_interval 200 --adjust_iterations 12000 --q_required "$q_required" \
  --start_checkpoint "/workspace/PowerGS/models/mini-splatting-d/m360/"$scene_name"/chkpnt30000.pth" --port 2222
  for i in "${!PSs[@]}"; do
      echo "Running iteration $((i+1)) of ${#PSs[@]}"
      python FR_Display/fr_display.py \
        -s "$m360_dataset_base_dir/$scene_name" \
        -m "/workspace/PowerGS/models/$method_name/m360/$scene_name/levels/L${save_levels[i]}" \
        -i images_4 \
        --eval \
        --checkpoint_iterations  "${save_CKPTs[i]}"\
        --adjust_interval 100 \
        --adjust_iterations 4500 \
        --start_checkpoint "/workspace/PowerGS/models/$method_name/m360/$scene_name/levels/L${start_levels[i]}/chkpnt${start_CKPTs[i]}.pth" \
        --port 2222 \
        --pooling_size "${PSs[i]}" \
        --level "${save_levels[i]}"
  done
elif is_in_list "$scene_name" "${mipnerf360_indoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 indoor"
  # For indoor scenes, you might follow exactly the same pattern, 
  # or if needed, adjust parameters/checkpoints (shown below is a direct copy).

  python FR_Display/fr_display_L1.py -s "$m360_dataset_base_dir/$scene_name" -m "/workspace/PowerGS/models/$method_name/m360/$scene_name/levels/L1" -i images_2 --eval \
  --checkpoint_iterations 45000 --adjust_interval 200 --adjust_iterations 12000 --q_required "$q_required" \
  --start_checkpoint "/workspace/PowerGS/models/mini-splatting-d/m360/"$scene_name"/chkpnt30000.pth" --port 2222
  for i in "${!PSs[@]}"; do
      echo "Running iteration $((i+1)) of ${#PSs[@]}"
      python FR_Display/fr_display.py \
        -s "$m360_dataset_base_dir/$scene_name" \
        -m "/workspace/PowerGS/models/$method_name/m360/$scene_name/levels/L${save_levels[i]}" \
        -i images_2 \
        --eval \
        --checkpoint_iterations  "${save_CKPTs[i]}"\
        --adjust_interval 100 \
        --adjust_iterations 4500 \
        --start_checkpoint "/workspace/PowerGS/models/$method_name/m360/$scene_name/levels/L${start_levels[i]}/chkpnt${start_CKPTs[i]}.pth" \
        --port 2222 \
        --pooling_size "${PSs[i]}" \
        --level "${save_levels[i]}"
  done

elif is_in_list "$scene_name" "${nerf_synthetic_scene[@]}"; then
  echo "Scene type: nerf_synthetic"
  # Here, the main difference is the base directory and possibly the model path.
  # Adjust as needed if your synthetic scenes use different default checkpoints or iteration values.

  python FR_Display/fr_display_L1.py -s "$nerf_synthetic_dataset_base_dir/$scene_name" -m "/workspace/PowerGS/models/$method_name/nerf_synthetic/$scene_name/levels/L1" -i images --eval \
  --checkpoint_iterations 45000 --adjust_interval 200 --adjust_iterations 12000 --q_required "$q_required" \
  --start_checkpoint "/workspace/PowerGS/models/3dgs/nerf_synthetic/"$scene_name"/chkpnt30000.pth" --port 2222
  for i in "${!PSs[@]}"; do
      echo "Running iteration $((i+1)) of ${#PSs[@]}"
      python FR_Display/fr_display.py \
        -s "$nerf_synthetic_dataset_base_dir/$scene_name" \
        -m "/workspace/PowerGS/models/$method_name/nerf_synthetic/$scene_name/levels/L${save_levels[i]}" \
        -i images \
        --eval \
        --checkpoint_iterations  "${save_CKPTs[i]}"\
        --adjust_interval 100 \
        --adjust_iterations 4500 \
        --start_checkpoint "/workspace/PowerGS/models/$method_name/nerf_synthetic/$scene_name/levels/L${start_levels[i]}/chkpnt${start_CKPTs[i]}.pth" \
        --port 2222 \
        --pooling_size "${PSs[i]}" \
        --level "${save_levels[i]}"
  done

elif is_in_list "$scene_name" "${extend_scenes[@]}"; then
  echo "Scene type: extend"
  # Here, the main difference is the base directory and possibly the model path.
  # Adjust as needed if your synthetic scenes use different default checkpoints or iteration values.
  python FR_Display/fr_display_L1.py -s "$extend_dataset_base_dir/$scene_name" -m "/workspace/PowerGS/models/$method_name/extend/$scene_name/levels/L1" -i images --eval \
  --checkpoint_iterations 45000 --adjust_interval 200 --adjust_iterations 12000 --q_required "$q_required" \
  --start_checkpoint "/workspace/PowerGS/models/3dgs/extend/"$scene_name"/chkpnt30000.pth" --port 2222
  for i in "${!PSs[@]}"; do
      echo "Running iteration $((i+1)) of ${#PSs[@]}"
      python FR_Display/fr_display.py \
        -s "$extend_dataset_base_dir/$scene_name" \
        -m "/workspace/PowerGS/models/$method_name/extend/$scene_name/levels/L${save_levels[i]}" \
        -i images \
        --eval \
        --checkpoint_iterations  "${save_CKPTs[i]}"\
        --adjust_interval 100 \
        --adjust_iterations 4500 \
        --start_checkpoint "/workspace/PowerGS/models/$method_name/extend/$scene_name/levels/L${start_levels[i]}/chkpnt${start_CKPTs[i]}.pth" \
        --port 2222 \
        --pooling_size "${PSs[i]}" \
        --level "${save_levels[i]}"
  done

else
  echo "Error: Scene name '$scene_name' does not match any known types."
  exit 1
fi