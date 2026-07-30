"""D3 gate: inference prompt == trainer completion prefix (byte-identical), and
full training text == stripped infer prompt + assistant JSON + im_end.

Mirrors mlx-vlm 0.6.7 trainer/datasets.py exactly:
  full text  = apply_chat_template(conv,      add_generation_prompt=False)
  mask prefix= apply_chat_template(conv[:-1], add_generation_prompt=True)
infer.py builds apply_chat_template(processor, config, record_text, num_images=N)
(= mask prefix) and strips the trailing '<think>\n' before generation.
"""
import argparse
import json
import sys
from pathlib import Path

from mlx_vlm.prompt_utils import apply_chat_template as tmpl
from mlx_vlm.utils import load_config
from transformers import AutoProcessor

from factored_pipeline.contract import IDENTIFY_PROMPT, IDENTIFY_PROMPT_V1

MODEL = str(Path("~/models/mlx-community/Qwen3.5-4B-MLX-4bit").expanduser())
REPO = Path(__file__).resolve().parent

processor = AutoProcessor.from_pretrained(MODEL)
config = load_config(MODEL)

def check(jsonl, n=5, expected_text=None):
    if not Path(jsonl).exists():
        print(f"  [SKIP] not present")
        return 0
    fails = 0
    with open(jsonl) as f:
        recs = [json.loads(l) for _, l in zip(range(n), f)]
    for rec in recs:
        conv = rec["messages"]
        user = conv[0]["content"]
        text = next(i["text"] for i in user if i["type"] == "text")
        n_img = sum(1 for i in user if i["type"] == "image")
        asst = conv[1]["content"][0]["text"]
        record_expected_text = expected_text
        if record_expected_text is None:
            try:
                completion = json.loads(asst)
                if set(completion) == {"not_food", "container", "items"}:
                    record_expected_text = IDENTIFY_PROMPT
            except (json.JSONDecodeError, TypeError):
                pass

        # trainer side (datasets.py process/_completion_prefix)
        full = tmpl(processor, config, conv, add_generation_prompt=False,
                    num_images=n_img)
        trainer_prefix = tmpl(processor, config, conv[:-1],
                              add_generation_prompt=True, num_images=n_img)

        # inference side (infer.run_inference)
        infer = tmpl(processor, config, text, num_images=n_img)
        infer_stripped = infer[:-len("<think>\n")] if infer.endswith("<think>\n") else infer

        ok0 = record_expected_text is None or text == record_expected_text
        ok1 = infer == trainer_prefix
        ok2 = full.startswith(infer_stripped + asst)
        tail = full[len(infer_stripped) + len(asst):]
        ok3 = tail in ("<|im_end|>\n", "<|im_end|>")
        status = "OK" if (ok0 and ok1 and ok2 and ok3) else (
            f"FAIL(text_eq={ok0},prefix_eq={ok1},full={ok2},tail={ok3} {tail!r})"
        )
        if status != "OK":
            fails += 1
        print(f"  [{status}] {n_img} img(s), depth_line={'depth sensor' in text}")
    return fails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path)
parser.add_argument("--records", type=int, default=5)
args = parser.parse_args()

paths = (
    [
        (
            "requested    ",
            args.data,
            (
                IDENTIFY_PROMPT_V1
                if "id_v1" in args.data.name or "id_v1" in args.data.parent.name
                else IDENTIFY_PROMPT
                if "id_" in args.data.name or "id_" in args.data.parent.name
                else None
            ),
        )
    ]
    if args.data
    else [
        ("v5 control   ", REPO / "finetune_data_v2/train.jsonl", None),
        ("variant B txt", REPO / "finetune_data_v2d_txt/train.jsonl", None),
        ("variant A img", REPO / "finetune_data_v2d_img/train.jsonl", None),
        # v7 not-food track: refusal records use the byte-identical v5 prompt
        # (only the completion differs), so they must pass the same gate.
        ("v7 negatives ", REPO / "negatives/train.jsonl", None),
        ("v7 mixed     ", REPO / "finetune_data_v7/train.jsonl", None),
        ("v8 numeric   ", REPO / "finetune_data_v8_numeric/train.jsonl", None),
        (
            "identify v1  ",
            REPO / "finetune_data_id_v1/train.jsonl",
            IDENTIFY_PROMPT_V1,
        ),
        (
            "identify v2  ",
            REPO / "finetune_data_id_v2/train.jsonl",
            IDENTIFY_PROMPT,
        ),
    ]
)
total = 0
for name, path, expected_text in paths:
    print(f"{name}: {path}")
    total += check(path, n=args.records, expected_text=expected_text)

print("\nPARITY GATE:", "PASSED" if total == 0 else f"FAILED ({total} records)")
sys.exit(1 if total else 0)
