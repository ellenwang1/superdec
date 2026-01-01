#!/bin/bash
set -euo pipefail  # stop on error, undefined variable, or pipeline failure

# trap to print the failing command
trap 'echo "Error occurred at command: $BASH_COMMAND"' ERR

ROBOT_NAME=fetch
OBJECTS_SCENE_DIR=data/robots/$ROBOT_NAME/ply
OUTPUT_NPZ_DIR=data/robots/$ROBOT_NAME/npz # path to the folder where to save the output .npz files
OUTPUT_SQ_DIR=data/robots/$ROBOT_NAME/superquadrics
Z_UP=true

python superdec/utils/urdf_to_ply.py \
    data/robots/$ROBOT_NAME/$ROBOT_NAME.urdf \
    $OBJECTS_SCENE_DIR \
    --package_root superdec \
    --points_per_link 5000

python superdec/utils/ply_to_npz.py \
    --input_path="$OBJECTS_SCENE_DIR" \
    --scene_name="$ROBOT_NAME" \
    --output_dir="$OUTPUT_NPZ_DIR"

python superdec/evaluate/to_npz.py \
    checkpoints_folder="checkpoints/normalized" \
    input_dir="$OUTPUT_NPZ_DIR" \
    output_dir="$OUTPUT_SQ_DIR" \
    dataset=scene \
    scene.name="$ROBOT_NAME" \
    scene.z_up="$Z_UP"

python superdec/visualization/object_visualizer.py \
    dataset=scene \
    split="$ROBOT_NAME" \
    npz_folder="$OUTPUT_SQ_DIR"