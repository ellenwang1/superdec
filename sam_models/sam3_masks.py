import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model


def run_sam3(image_path: str, scene_id: str, prompts: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    mask_dir = os.path.join(ROOT, "data", "raw", f"{scene_id}_masks")
    os.makedirs(mask_dir, exist_ok=True)

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    mask_id = 0

    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        for mask in results["masks"]:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            colored = img_np * mask_np[:, :, None]

            out_path = os.path.join(mask_dir, f"{mask_id}.png")
            Image.fromarray(colored).save(out_path)
            mask_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)

    args = parser.parse_args()

    run_sam3(
        image_path=args.image_path,
        scene_id=args.scene_id,
        prompts=args.prompts,
    )


if __name__ == "__main__":
    main()
