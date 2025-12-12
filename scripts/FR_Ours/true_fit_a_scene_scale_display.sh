#!/bin/bash

# Input scene name
scene_name=$1
q_required=$2
port=$3
scale=$4

PSs=(3 7 12)
start_levels=( 1 2 3 )
save_levels=( 2 3 4 )
start_CKPTs=( 60000 65000 70000 )
save_CKPTs=( 65000 70000 75000 )
# Export paths
export m360_dataset_base_dir=/workspace/dataset/m360
export nerf_synthetic_dataset_base_dir=/workspace/dataset/nerf_synthetic
export code_dir=/workspace/PowerGS/
export method_name="FR_Ours_"$q_required"_"$scale""

# Scene type lists
mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
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

cd $code_dir

dataset_base=0
data_type=0
image_type=0
dense_model=0
# Determine the scene type and run the appropriate command
if is_in_list "$scene_name" "${mipnerf360_outdoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 outdoor"
  dataset_base=$m360_dataset_base_dir
  data_type="m360" 
  image_type="images_4"   
  dense_model="mini-splatting-d"

else
  echo "Error: Scene name '$scene_name' does not match any known types."
  exit 1
fi

# first perform coarse-to-fine sampling
python FR_Ours/fr_prune_L1.py -s "$dataset_base/$scene_name" -m "/workspace/PowerGS/models/$method_name/$data_type/$scene_name/levels/L1" -i $image_type --eval \
    --q_required "$q_required" \
    --start_checkpoint "/workspace/PowerGS/models/"$dense_model"/$data_type/"$scene_name"/chkpnt30000.pth" --port $port \
    --display_scale $scale

# then curve fitting 
python3 FR_Ours/curve_fit_scene.py --root_dir /workspace/PowerGS/models/"$method_name"/"$data_type"/"$scene_name"/levels/ --level 1

# then perform final pruning
python FR_Ours/final_prune_L1.py -s "$dataset_base/$scene_name" -m "/workspace/PowerGS/models/$method_name/"$data_type"/$scene_name/levels/L1" -i $image_type --eval \
    --checkpoint_iterations 60000 --prune_interval 1000 --prune_iterations 15000 --q_required "$q_required" --adjust_interval 200 --adjust_iterations 10000 \
    --start_checkpoint "/workspace/PowerGS/models/"$dense_model"/"$data_type"/"$scene_name"/chkpnt30000.pth" --port $port


for i in "${!PSs[@]}"; do
    echo "Running iteration $((i+1)) of ${#PSs[@]}"
    # search for the best mask ratio
    python FR_Ours/fr_mask.py \
    -s "$dataset_base/$scene_name" \
    -m "/workspace/PowerGS/models/$method_name/$data_type/$scene_name/levels/L${save_levels[i]}" \
    -i  $image_type \
    --eval \
    --start_checkpoint "/workspace/PowerGS/models/$method_name/$data_type/$scene_name/levels/L${start_levels[i]}/chkpnt${start_CKPTs[i]}.pth" \
    --port $port \
    --pooling_size "${PSs[i]}" \
    --level "${save_levels[i]}" \
    --display_scale $scale

    python3 FR_Ours/curve_fit_scene.py --root_dir /workspace/PowerGS/models/"$method_name"/"$data_type"/"$scene_name"/levels/ --level "${save_levels[i]}"

    # then perform final pruning
    python FR_Ours/fr_final_mask.py \
      -s "$dataset_base/$scene_name" \
      -m "/workspace/PowerGS/models/$method_name/"$data_type"/$scene_name/levels/L${save_levels[i]}" \
      -i $image_type \
      --eval \
      --checkpoint_iterations "${save_CKPTs[i]}" \
      --adjust_interval 50 \
      --adjust_iterations 1100 \
      --prune_interval 200 \
      --prune_iterations 3400 \
      --start_checkpoint "/workspace/PowerGS/models/$method_name/"$data_type"/$scene_name/levels/L${start_levels[i]}/chkpnt${start_CKPTs[i]}.pth" \
      --port $port \
      --pooling_size "${PSs[i]}" \
      --level "${save_levels[i]}"

done