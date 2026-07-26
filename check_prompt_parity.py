"""D3 gate: inference prompt == trainer completion prefix (byte-identical), and
full training text == stripped infer prompt + assistant JSON + im_end.

Mirrors mlx-vlm 0.6.7 trainer/datasets.py exactly:
  full text  = apply_chat_template(conv,      add_generation_prompt=False)
  mask prefix= apply_chat_template(conv[:-1], add_generation_prompt=True)
03_infer.py builds apply_chat_template(processor, config, record_text, num_images=N)
(= mask prefix) and strips the trailing '<think>\n' before generation.
"""
import json
import sys
from pathlib import Path

from mlx_vlm.prompt_utils import apply_chat_template as tmpl
from mlx_vlm.utils import load_config
from transformers import AutoProcessor

MODEL = "/Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit"
REPO = Path("/Users/jevgenikabanov/Documents/Projects/Claude/SeeCal")

processor = AutoProcessor.from_pretrained(MODEL)
config = load_config(MODEL)

def check(jsonl, n=5):
    fails = 0
    with open(jsonl) as f:
        recs = [json.loads(l) for _, l in zip(range(n), f)]
    for rec in recs:
        conv = rec["messages"]
        user = conv[0]["content"]
        text = next(i["text"] for i in user if i["type"] == "text")
        n_img = sum(1 for i in user if i["type"] == "image")
        asst = conv[1]["content"][0]["text"]

        # trainer side (datasets.py process/_completion_prefix)
        full = tmpl(processor, config, conv, add_generation_prompt=False,
                    num_images=n_img)
        trainer_prefix = tmpl(processor, config, conv[:-1],
                              add_generation_prompt=True, num_images=n_img)

        # inference side (03_infer.run_inference)
        infer = tmpl(processor, config, text, num_images=n_img)
        infer_stripped = infer[:-len("<think>\n")] if infer.endswith("<think>\n") else infer

        ok1 = infer == trainer_prefix
        ok2 = full.startswith(infer_stripped + asst)
        tail = full[len(infer_stripped) + len(asst):]
        ok3 = tail in ("<|im_end|>\n", "<|im_end|>")
        status = "OK" if (ok1 and ok2 and ok3) else f"FAIL(prefix_eq={ok1},full={ok2},tail={ok3} {tail!r})"
        if status != "OK":
            fails += 1
        print(f"  [{status}] {n_img} img(s), depth_line={'depth sensor' in text}")
    return fails

total = 0
for name, path in [
    ("v5 control   ", REPO / "finetune_data_v2/train.jsonl"),
    ("variant B txt", REPO / "finetune_data_v2d_txt/train.jsonl"),
    ("variant A img", REPO / "finetune_data_v2d_img/train.jsonl"),
]:
    print(f"{name}: {path}")
    total += check(path)

print("\nPARITY GATE:", "PASSED" if total == 0 else f"FAILED ({total} records)")
sys.exit(1 if total else 0)
