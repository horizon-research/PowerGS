#!/bin/bash

# Input scene name
scene_name=$1
method_name=$2

# Export paths
export m360_dataset_base_dir=/workspace/dataset/m360
export nerf_synthetic_dataset_base_dir=/workspace/dataset/nerf_synthetic
export render_code_dir=/workspace/PowerGS/ # use PowerGS metric protocol
export metric_code_dir=/workspace/PowerGS/ # use PowerGS metric protocol
export output_path="/workspace/PowerGS/models/$method_name"

export extend_dataset_base_dir="/workspace/dataset/extend"

extend_scenes=("drjohnson" "playroom" "train" "truck")
# Scene type lists
mipnerf360_outdoor_scenes=("flowers" "bicycle" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
nerf_synthetic_scene=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

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


# Determine the scene type and execute the appropriate commands
if is_in_list "$scene_name" "${mipnerf360_outdoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 outdoor"
  dataset_path="$m360_dataset_base_dir/$scene_name"
  model_path="$output_path/m360/$scene_name"/levels/
  imp_metric="outdoor"
elif is_in_list "$scene_name" "${mipnerf360_indoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 indoor"
  dataset_path="$m360_dataset_base_dir/$scene_name"
  model_path="$output_path/m360/$scene_name"/levels/
  imp_metric="indoor"
elif is_in_list "$scene_name" "${nerf_synthetic_scene[@]}"; then
  echo "Scene type: nerf synthetic"
  dataset_path="$nerf_synthetic_dataset_base_dir/$scene_name"
  model_path="$output_path/nerf_synthetic/$scene_name"/levels/
  imp_metric="indoor"
else
  echo "Scene type: extend"
  dataset_path="$extend_dataset_base_dir/$scene_name"
  model_path="$output_path/extend/$scene_name"/levels/
  imp_metric="outdoor"
fi


# Render (if skip_rendering is not set)

cd $render_code_dir
python render_L1.py --iteration 35000 -s "$dataset_path" -m "$model_path"/L1 --quiet --eval --skip_train --FR_folder "$model_path"
python3 test_power_FR.py --iteration 30000 -s "$dataset_path" -m "$model_path"/L1 --eval --skip_train --FR_folder "$model_path"
python metrics.py -m "$model_path"/L1
python metrics_FR.py -m "$model_path"


