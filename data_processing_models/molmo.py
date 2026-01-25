import argparse
import json
import os
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_ID = "allenai/Molmo2-8B"


def load_molmo(model_id: str = MODEL_ID, device_map: str = "cpu"):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map,
    )
    model.eval()
    return processor, model


def parse_unique_lines(text: str) -> List[str]:
    """Deduplicate while preserving order."""
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


@torch.inference_mode()
def run_molmo(
    image_path: str,
    scene_id: str,
    prompts: List[str],
    output_json: str,
    model_id: str = MODEL_ID,
    device_map: str = "cpu",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    # Load model once per run
    processor, model = load_molmo(model_id=model_id, device_map=device_map)

    # Load image
    image = Image.open(image_path).convert("RGB")

    results = []
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
        generated_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        results.append(
            {
                "prompt": prompt,
                "raw_text": generated_text,
                "items": parse_unique_lines(generated_text),  # list for JSON
            }
        )

    payload = {
        "scene_id": scene_id,
        "image_path": image_path,
        "model_id": model_id,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--device_map", default="cpu")  # keep CPU default
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    output_json = args.output_json or f"outputs/molmo_{args.scene_id}.json"

    run_molmo(
        image_path=args.image_path,
        scene_id=args.scene_id,
        prompts=args.prompts,
        output_json=output_json,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()

