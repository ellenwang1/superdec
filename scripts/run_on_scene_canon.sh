#!/bin/bash
set -euo pipefail
trap 'echo "Error at: $BASH_COMMAND"' ERR

# ------------------ GLOBAL CONFIG ------------------
SCENE_ID=9521
Z_UP=true
TAG=hf

# Text prompts for SAM3
PROMPTS=("Cup" "Box" "Chair" "Bookshelf")

# ------------------ PATHS ------------------
RAW_IMAGE=data/raw/$SCENE_ID.jpeg

CANON_OBJ=data/recon/$SCENE_ID/canon/objects
WORLD_OBJ=data/recon/$SCENE_ID/world/objects
TRANSFORMS=data/transforms/$SCENE_ID/canon_to_world

PC_CANON=data/processed/$SCENE_ID/pc_npz/canon
PC_WORLD=data/processed/$SCENE_ID/pc_npz/world

SQ_CANON=data/processed/$SCENE_ID/superquadrics/canon
SQ_WORLD=data/processed/$SCENE_ID/superquadrics/world

VIZ_DIR=data/viz/$SCENE_ID/superquadrics

# ------------------ STAGE 0: SANITY CHECK ------------------
echo "Scene: $SCENE_ID"
echo "Image path: $RAW_IMAGE"

if [ ! -f "$RAW_IMAGE" ]; then
  echo "Image file not found: $RAW_IMAGE"
  exit 1
fi

# ------------------ STAGE 1: SAM3 MASKS ------------------
echo "Stage 1: SAM3 mask generation"

# Single image path
IMAGE_PATH=data/raw/$SCENE_ID.jpeg

# Mask directory
MASK_DIR=data/raw/${SCENE_ID}
python sam_models/sam3_masks.py \
    --scene_id "$SCENE_ID" \
    --image_path "$IMAGE_PATH" \
    --prompts "${PROMPTS[@]}"


# ------------------ STAGE 2: WORLD RECONSTRUCTION ------------------
echo "Stage 2: World-space reconstruction"

WORLD_OUT=data/recon/$SCENE_ID/world/objects/$SCENE_ID

python sam_models/run_sam3d_multi_world.py \
    --scene_id "$SCENE_ID" \
    --image_path "$RAW_IMAGE" \
    --tag "$TAG"


# ------------------ STAGE 3: CANONICAL RECONSTRUCTION ------------------
echo "Stage 3: Canonical reconstruction"

CANON_OUT=data/recon/$SCENE_ID/canon/objects/$SCENE_ID

python sam_models/run_sam3d_single_canonical.py \
    --scene_id "$SCENE_ID" \
    --image_path "$RAW_IMAGE" \
    --tag "$TAG"

# ------------------ STAGE 4: PLY TO NPZ (CANONICAL) ------------------
echo "Stage 4: Canonical PLY to NPZ"

python superdec/utils/ply_to_npz.py \
  --input_path "$CANON_OBJ" \
  --scene_name "$SCENE_ID" \
  --output_dir "$PC_CANON"

# ------------------ STAGE 5: CANONICAL TO WORLD TRANSFORMS ------------------
echo "Stage 5: Canonical to world transforms"

python superdec/utils/canon_to_world_transforms.py \
  --input_canon_path "$CANON_OBJ" \
  --input_world_path "$WORLD_OBJ" \
  --output_dir "$TRANSFORMS"

# ------------------ STAGE 6: SUPERQUADRICS (CANONICAL) ------------------
echo "Stage 6: Superquadric fitting (canonical)"

python superdec/evaluate/to_npz.py \
  checkpoints_folder="checkpoints/normalized" \
  dataset=scene \
  scene.name="'$SCENE_ID'" \
  scene.z_up="$Z_UP" \
  input_dir="$PC_CANON" \
  output_dir="$SQ_CANON" 

# ------------------ STAGE 7: MAP SUPERQUADRICS TO WORLD ------------------
echo "Stage 7: Canonical to world superquadrics"

python superdec/utils/canon_to_world.py \
  --npz_folder "$SQ_CANON" \
  --transforms_folder "$TRANSFORMS" \
  --output_folder "$SQ_WORLD"

# ------------------ STAGE 8: VISUALIZATION ------------------
echo "Stage 8: Visualization"

python superdec/visualization/object_visualizer.py \
  dataset=scene \
  split="$SCENE_ID" \
  npz_folder="$SQ_WORLD" \

echo "Full pipeline complete"
