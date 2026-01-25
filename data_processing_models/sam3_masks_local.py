import os
import argparse
import torch
import json
import numpy as np
from PIL import Image

from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor


def run_sam3(image_path: str, scene_id: str, objects: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    mask_dir = os.path.join(ROOT, "data", "raw", f"{scene_id}_masks")
    os.makedirs(mask_dir, exist_ok=True)

    # ----------------------------
    # Load local SAM3 model
    # ----------------------------
    bpe_path = os.path.join(ROOT, "data_processing_models", "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path).to(device)
    model.eval()

    processor = Sam3Processor(model)

    # ----------------------------
    # Load image
    # ----------------------------
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    # Initialize image state once
    inference_state = processor.set_image(image)

    mask_id = 0

    print(f"List of objects: {objects}")
    for object in objects:
        # ----------------------------
        # Text prompt inference
        # ----------------------------
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=object,
        )

        masks = output["masks"]  # (N, H, W) torch.Tensor

        for mask in masks:
            mask_np = mask.detach().cpu().numpy()
            
            # Normalise 
            mask_np = np.squeeze(mask_np)
            mask_np = (mask_np > 0).astype(np.uint8)

            # Match original behavior: masked RGB image
            colored = img_np * mask_np[:, :, None]

            out_path = os.path.join(mask_dir, f"{mask_id}.png")
            Image.fromarray(colored).save(out_path)
            mask_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_json", required=True)

    args = parser.parse_args()

    with open(args.output_json) as f:
        data = json.load(f)

    objects_list = data["results"][0]["items"]

    run_sam3(
        image_path=args.image_path,
        scene_id=args.scene_id,
        objects=objects_list,
    )


if __name__ == "__main__":
    main()
