import os
import argparse
from inference import (
    Inference,
    load_image,
    load_masks,
    make_scene_no_merge,
    ply_to_png,
)


def run_world_single_image(image_path: str, scene_id: str, tag: str):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Mask directory: assume same folder as image with "_masks" suffix
    mask_dir = os.path.join(os.path.dirname(image_path), f"{scene_id}_masks")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    # Output directories
    world_root = os.path.join(ROOT, "data", "recon", scene_id, "world")
    obj_dir = os.path.join(world_root, "objects")
    png_dir = os.path.join(world_root, "previews")

    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print(f"[WORLD] Scene: {scene_id}")
    print(f"[WORLD] Image path: {image_path}")
    print(f"[WORLD] Output dir: {obj_dir}")

    # Inference setup
    config_path = os.path.join(ROOT, "checkpoints", tag, "pipeline.yaml")
    inference = Inference(config_path, compile=False)

    # Load image and masks
    image = load_image(image_path)
    masks = load_masks(mask_dir, extension=".png")

    if len(masks) == 0:
        raise RuntimeError("No masks found")

    # Inference per mask
    outputs = []
    for i, mask in enumerate(masks):
        print(f"[WORLD] Inference on mask {i}")
        outputs.append(inference(image, mask, seed=42))

    scenes = make_scene_no_merge(*outputs)

    # Save outputs
    for i, gs in enumerate(scenes):
        ply_path = os.path.join(obj_dir, f"{i}.ply")
        png_path = os.path.join(png_dir, f"{i}.png")

        gs.save_ply(ply_path)
        ply_to_png(ply_path, png_path)

        print(f"[WORLD] Saved {ply_path}")

    print("[WORLD] Done.")


def main():
    parser = argparse.ArgumentParser(description="World-space reconstruction for one image")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--scene_id", required=True, help="Scene ID")
    parser.add_argument("--tag", default="hf", help="Tag for checkpoint folder")

    args = parser.parse_args()

    run_world_single_image(
        image_path=args.image_path,
        scene_id=args.scene_id,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
