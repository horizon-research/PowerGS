#!/bin/bash
quality_required=$1

# Scene type lists
mipnerf360_outdoor_scenes=( )
# mipnerf360_outdoor_scenes=("flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=( )
nerf_synthetic_scene=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# nerf_synthetic_scene=("chair" "drums" "ficus" "hotdog" "materials" "mic" "ship")

# Function to train scenes
function train_scenes() {
    local scene_list=("$@")
    for scene in "${scene_list[@]}"; do
        echo "Training scene: $scene"
        bash train_a_scene.sh "$scene" "$quality_required"
        if [[ $? -ne 0 ]]; then
            echo "Error occurred while training $scene. Exiting."
            exit 1
        fi
    done
}

# Train mipnerf360 outdoor scenes
echo "Training mipnerf360 outdoor scenes..."
train_scenes "${mipnerf360_outdoor_scenes[@]}"

# Train mipnerf360 indoor scenes
echo "Training mipnerf360 indoor scenes..."
train_scenes "${mipnerf360_indoor_scenes[@]}"

# Train nerf synthetic scenes
echo "Training nerf synthetic scenes..."
train_scenes "${nerf_synthetic_scene[@]}"

echo "All scenes have been successfully trained!"
