"""
03_infer.py
-----------
Run inference on food images using the fine-tuned Qwen3.5-4B model with
LoRA adapters via mlx-vlm.

This avoids the need to fuse adapters — mlx-vlm loads the base model and
applies LoRA weights on the fly.

Usage:
  # Single image
  python 03_infer.py --image dataset_clean/dish_1565370004/overhead.jpg

  # Multiple images
  python 03_infer.py --image img1.jpg img2.jpg img3.jpg

  # Evaluate on test set (first N samples)
  python 03_infer.py --test-set finetune_data/test.jsonl --limit 20

  # Custom adapter/model paths
  python 03_infer.py --image food.jpg \
    --model-path ~/models/Qwen3.5-4B-MLX-bf16 \
    --adapter-path adapters
"""

import argparse
import json
import sys
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template


# Same prompts used during training (must match 02_prepare_finetune.py)
SYSTEM_PROMPT = (
    "You are a nutrition expert. When shown a photo of a meal, "
    "you identify the ingredients with their weights and estimate the total "
    "nutritional content with high accuracy. Always respond with a valid JSON object."
)

USER_PROMPT = (
    "Look at this meal and identify its ingredients and nutritional content. "
    "Provide your answer as a JSON object with the following keys: "
    "total_calories (kcal), protein_g, fat_g, carbs_g, and items "
    "(a list of objects with name, estimated_grams, calories, protein_g, fat_g, carbs_g, "
    "sorted by weight descending)."
)


def run_inference(model, processor, image_path: str, max_tokens: int = 512,
                  config=None, debug: bool = False,
                  thinking_budget: int = None) -> dict:
    """Run inference on a single image and return parsed JSON.

    Training format has NO system message — the system instruction is folded
    into the user message text (required to satisfy both pyarrow's uniform
    schema constraint and Qwen3.5's Jinja2 template constraint).  Inference
    must match: system instruction prepended to USER_PROMPT, no system block.
    """
    # Combine system instruction + user question, matching training exactly.
    combined_prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT
    prompt = mlx_apply_chat_template(
        processor,
        config,
        combined_prompt,
        num_images=1,
    )
    # prompt is: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n

    # Thinking mode: if thinking_budget is set, keep <think>\n in the prompt
    # and let the model reason before producing JSON. Otherwise strip it so
    # the model outputs JSON directly (faster, matches training format).
    if thinking_budget is None:
        if prompt.endswith("<think>\n"):
            prompt = prompt[:-len("<think>\n")]

    if debug:
        print(f"\n[DEBUG] Prompt ({len(prompt)} chars):\n{repr(prompt[:800])}\n")
        print(f"[DEBUG] thinking_budget={thinking_budget}\n")

    generate_kwargs = dict(
        temperature=0.1,
        repetition_penalty=1.1,
        verbose=False,
    )
    if thinking_budget is not None:
        generate_kwargs["enable_thinking"] = True
        generate_kwargs["thinking_budget"] = thinking_budget
        generate_kwargs["thinking_start_token"] = "<think>"
        generate_kwargs["thinking_end_token"] = "</think>"

    output = generate(
        model,
        processor,
        prompt=prompt,
        image=image_path,
        max_tokens=max_tokens,
        **generate_kwargs,
    )

    # GenerationResult object (mlx-vlm >= 0.x) — extract text
    if hasattr(output, "text"):
        output = output.text
    elif hasattr(output, "generation"):
        output = output.generation
    else:
        output = str(output)

    # Strip thinking block <think>...</think> before JSON parsing
    import re as _re
    output = _re.sub(r"<think>.*?</think>", "", output, flags=_re.DOTALL).strip()

    if debug and thinking_budget is not None:
        print(f"[DEBUG] Output after stripping thinking block:\n{output[:500]}\n")

    # Try to parse JSON from the output
    try:
        result = json.loads(output.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from the output (model might add extra text)
        import re
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = {"raw_output": output, "parse_error": True}
        else:
            result = {"raw_output": output, "parse_error": True}

    return result


def evaluate_test_set(model, processor, test_jsonl: Path, limit: int, data_dir: Path,
                      config=None, thinking_budget: int = None,
                      max_tokens: int = 1536, out_json: Path = None):
    """Evaluate on test set and compute metrics."""
    records = []
    with test_jsonl.open() as f:
        for line in f:
            records.append(json.loads(line))

    if limit:
        records = records[:limit]

    print(f"\nEvaluating on {len(records)} test samples...\n")

    errors_cal = []
    errors_protein = []
    errors_fat = []
    errors_carbs = []
    parse_failures = 0
    schema_failures = 0

    for i, rec in enumerate(records):
        # Get ground truth from assistant message
        # content is [{"type": "text", "text": "{...json...}"}]
        asst_content = rec["messages"][-1]["content"]
        if isinstance(asst_content, list):
            asst_text = asst_content[0]["text"]
        else:
            asst_text = asst_content
        gt = json.loads(asst_text)

        # Resolve image path: lives in user message content array
        user_content = rec["messages"][0]["content"]
        image_rel = next(item["image"] for item in user_content if item["type"] == "image")
        image_path = str((data_dir / image_rel).resolve())

        print(f"[{i+1}/{len(records)}] {Path(image_path).parent.name}/{Path(image_path).name}", end="")

        pred = run_inference(model, processor, image_path, max_tokens=max_tokens,
                             config=config, thinking_budget=thinking_budget)

        if pred.get("parse_error"):
            parse_failures += 1
            print(f" — PARSE ERROR")
            continue

        # Valid JSON but wrong/missing/non-numeric keys must not kill the run:
        # count separately so MAE is only over schema-conforming predictions.
        try:
            cal_err = abs(float(pred["total_calories"]) - float(gt["total_calories"]))
            prot_err = abs(float(pred["protein_g"]) - float(gt["protein_g"]))
            fat_err = abs(float(pred["fat_g"]) - float(gt["fat_g"]))
            carb_err = abs(float(pred["carbs_g"]) - float(gt["carbs_g"]))
        except (KeyError, TypeError, ValueError) as e:
            schema_failures += 1
            print(f" — SCHEMA ERROR ({e})")
            continue

        errors_cal.append(cal_err)
        errors_protein.append(prot_err)
        errors_fat.append(fat_err)
        errors_carbs.append(carb_err)

        print(f" — cal err: {cal_err:.1f} kcal  |  pred: {float(pred['total_calories']):.0f}  gt: {float(gt['total_calories']):.0f}")

    print(f"\n{'='*60}")
    print(f"Results on {len(records)} samples "
          f"({parse_failures} parse failures, {schema_failures} schema failures):")
    print(f"{'='*60}")

    import statistics

    def stats(name, errs):
        if not errs:
            print(f"  {name}: no valid predictions")
            return None
        mean = statistics.mean(errs)
        median = statistics.median(errs)
        print(f"  {name}: MAE={mean:.1f}  median={median:.1f}  (n={len(errs)})")
        return {"mae": mean, "median": median, "n": len(errs)}

    summary = {
        "samples": len(records),
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "calories": stats("Calories (kcal)", errors_cal),
        "protein_g": stats("Protein (g)", errors_protein),
        "fat_g": stats("Fat (g)", errors_fat),
        "carbs_g": stats("Carbs (g)", errors_carbs),
        "errors_cal": errors_cal,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {out_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SeeCal inference with fine-tuned Qwen3.5.")
    parser.add_argument(
        "--image", nargs="+", type=str, default=None,
        help="Path(s) to food image(s) for inference.",
    )
    parser.add_argument(
        "--test-set", type=Path, default=None,
        help="Path to test.jsonl for evaluation.",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Max samples to evaluate from test set (default: 20).",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="/Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit",
        help="Path to base model.",
    )
    parser.add_argument(
        "--adapter-path", type=str,
        default=None,
        help="Path to LoRA adapter directory. Omit when using a fused model.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1536,
        help="Max tokens to generate (p90 ground-truth JSON is ~700 tokens; "
             "512 truncates complex dishes and biases eval).",
    )
    parser.add_argument(
        "--out-json", type=Path, default=None,
        help="Write eval summary metrics to this JSON file (test-set mode).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print the formatted prompt for debugging.",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=None,
        help=(
            "Enable thinking mode with the given token budget (e.g. 512). "
            "Model reasons in a <think>…</think> block before producing JSON. "
            "Omit to disable thinking (faster, direct JSON output)."
        ),
    )
    args = parser.parse_args()

    if not args.image and not args.test_set:
        parser.error("Provide --image for single inference or --test-set for evaluation.")

    model_path = str(Path(args.model_path).expanduser())
    adapter_path = args.adapter_path

    print(f"Loading model: {model_path}")

    load_kwargs = {}
    if adapter_path:
        adapter_path_resolved = str(Path(adapter_path).expanduser())
        if Path(adapter_path_resolved).exists():
            print(f"Loading adapter: {adapter_path_resolved}")
            load_kwargs["adapter_path"] = adapter_path_resolved
        else:
            print(f"Note: adapter path '{adapter_path}' not found — loading base/fused model only.")
    else:
        print("No adapter path specified — loading base/fused model only.")

    model, processor = load(model_path, **load_kwargs)
    config = load_config(model_path)
    print("Model loaded.\n")

    if args.image:
        for img_path in args.image:
            print(f"Image: {img_path}")
            result = run_inference(model, processor, img_path, args.max_tokens, config,
                                   debug=args.debug,
                                   thinking_budget=args.thinking_budget)
            print(json.dumps(result, indent=2))
            print()

    if args.test_set:
        # Image paths in JSONL are relative to the project root (SeeCal/),
        # not relative to finetune_data/ — see 02_prepare_finetune.py image_str().
        data_dir = Path(__file__).parent.resolve()
        evaluate_test_set(model, processor, args.test_set, args.limit, data_dir, config,
                          thinking_budget=args.thinking_budget,
                          max_tokens=args.max_tokens, out_json=args.out_json)


if __name__ == "__main__":
    main()
