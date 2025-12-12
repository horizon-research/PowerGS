scene_name=$1
method_name=$2

# Scene type lists
mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
nerf_synthetic_scene=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

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
cd /workspace/PowerGS
# Determine the scene type and execute the appropriate commands
if is_in_list "$scene_name" "${mipnerf360_outdoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 outdoor"
  scene_type="m360"
elif is_in_list "$scene_name" "${mipnerf360_indoor_scenes[@]}"; then
  echo "Scene type: mipnerf360 indoor"
  scene_type="m360"
elif is_in_list "$scene_name" "${nerf_synthetic_scene[@]}"; then
  echo "Scene type: nerf synthetic"
  scene_type="nerf_synthetic"
else
  echo "Error: Scene name '$scene_name' does not match any known types."
  exit 1
fi

scene_folder="C:\Users\linwe\PowerGSs\dataset\"$scene_name"\levels"

python3 compose_FR.py --folder "$scene_folder"