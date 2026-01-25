import os
import argparse
import numpy as np
import torch
from inference import (
    Inference,
    load_image,
    load_masks,
    ply_to_png,
)


def run_canon_single_image(image_path: str, scene_id: str, tag: str):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Image name from file
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Mask directory: based on scene_id
    mask_dir = os.path.join("data", "raw", f"{scene_id}_masks")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    # Output directories
    canon_root = os.path.join(ROOT, "data", "recon", scene_id, "canon")
    obj_dir = os.path.join(canon_root, "objects")
    png_dir = os.path.join(canon_root, "previews")
    tfm_dir = os.path.join(canon_root, "transforms")

    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tfm_dir, exist_ok=True)

    print(f"[CANON] Scene: {scene_id}")
    print(f"[CANON] Image path: {image_path}")
    print(f"[CANON] Output dir: {obj_dir}")

    # Inference setup
    config_path = os.path.join(ROOT, "checkpoints", tag, "pipeline.yaml")
    inference = Inference(config_path, compile=False)

    # Load image and masks
    image = load_image(image_path)
    masks = load_masks(mask_dir, extension=".png")

    if len(masks) == 0:
        raise RuntimeError("No masks found")

    # Inference per mask
    for i, mask in enumerate(masks):
        print(f"[CANON] Inference on mask {i}")

        output = inference(image, mask, seed=42)

        transform = {
            "rotation": output["rotation"],
            "translation": output["translation"],
            "scale": output["scale"],
        }

        transform_np = {}
        for k, v in transform.items():
            if isinstance(v, torch.Tensor):
                transform_np[k] = v.detach().cpu().numpy()
            else:
                transform_np[k] = np.array(v)

        ply_path = os.path.join(obj_dir, f"{i}.ply")
        png_path = os.path.join(png_dir, f"{i}.png")
        npz_path = os.path.join(tfm_dir, f"{i}.npz")

        output["gs"].save_ply(ply_path)
        ply_to_png(ply_path, png_path)
        np.savez(npz_path, **transform_np)

        print(f"[CANON] Saved {ply_path}")
        print(f"[CANON] Saved {npz_path}")

    print("[CANON] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Canonical-space reconstruction for one image"
    )
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--scene_id", required=True, help="Scene ID")
    parser.add_argument("--tag", default="hf", help="Tag for checkpoint folder")

    args = parser.parse_args()

    run_canon_single_image(
        image_path=args.image_path,
        scene_id=args.scene_id,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
