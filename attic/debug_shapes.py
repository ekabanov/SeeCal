"""Diagnostic script: print pixel_values and input_ids shapes from dataset processing."""
import sys
sys.path.insert(0, "/sessions/sleepy-eloquent-archimedes/mnt/SeeCal/.venv/lib/python3.14/site-packages")

import json
import numpy as np

# Patch mlx so we can import mlx_vlm without a full MLX environment
try:
    import mlx.core as mx
except Exception:
    print("WARNING: mlx not importable, aborting")
    sys.exit(1)

from mlx_vlm.trainer.datasets import VisionDataset, get_prompt
from mlx_vlm.utils import prepare_inputs

# Load processor same way training does
from mlx_vlm.utils import load
model_path = "/Users/jevgenikabanov/models/Qwen3.5-4B-MLX-bf16"
print(f"Loading processor from {model_path}...")
model, processor = load(model_path)
config = model.config.__dict__ if hasattr(model.config, "__dict__") else {}

# Load one sample from train.jsonl
data_path = "/sessions/sleepy-eloquent-archimedes/mnt/SeeCal/finetune_data/train.jsonl"
print(f"Reading from {data_path}...")
with open(data_path) as f:
    sample_raw = json.loads(f.readline())

print("\n=== RAW SAMPLE KEYS ===")
print(list(sample_raw.keys()))

# Convert to datasets-style format
import datasets as hf_datasets
from datasets import Dataset
ds = Dataset.from_list([sample_raw] * 4)  # simulate batch of 4
print(f"\nDataset schema: {ds.features}")

# Replicate what mlx_vlm does: load via HuggingFace datasets (which goes through PyArrow)
sample_from_ds = ds[0]
print("\n=== SAMPLE FROM HF DATASET ===")
msgs = sample_from_ds.get("messages", [])
print(f"Number of messages: {len(msgs)}")
for i, msg in enumerate(msgs):
    role = msg.get("role", "?")
    content = msg.get("content", [])
    print(f"  msg[{i}] role={role}, content len={len(content)}")
    for j, item in enumerate(content):
        print(f"    item[{j}]: {item}")

# Use the mlx_vlm config dict
model_config = {}
if hasattr(model, "config"):
    cfg = model.config
    model_config["model_type"] = getattr(cfg, "model_type", "qwen3_5")
    # Get image_token_index
    model_config["image_token_index"] = getattr(cfg, "image_token_index", None)
    if model_config["image_token_index"] is None:
        model_config["image_token_index"] = getattr(cfg, "image_token_id", None)
    if hasattr(cfg, "__dict__"):
        model_config.update(cfg.__dict__)

print(f"\nmodel_type: {model_config.get('model_type')}")
print(f"image_token_index: {model_config.get('image_token_index')}")

# Now simulate VisionDataset.process for one item
from mlx_vlm.trainer.datasets import _extract_images_from_content, _normalize_conversation

item = sample_from_ds
images = item.get("images", item.get("image", []))
if not isinstance(images, list):
    images = [images] if images else []

conversations = item.get("messages", item.get("conversations"))
print(f"\nTop-level images count: {len(images)}")
print(f"Conversations type: {type(conversations)}")

# Extract images from content if none at top level
if not images and conversations is not None:
    conv_to_pass = conversations if isinstance(conversations[0], dict) else conversations[0] \
        if isinstance(conversations, list) and conversations else []
    images = _extract_images_from_content(conv_to_pass)
    print(f"Extracted {len(images)} images from content")
    for i, img in enumerate(images):
        print(f"  img[{i}]: {img.size} mode={img.mode}")

model_type = model_config.get("model_type")
prompts = []
if isinstance(conversations, list) and isinstance(conversations[0], list):
    for conv in conversations:
        prompt = get_prompt(model_type, processor, conv)
        prompts.append(prompt)
else:
    prompt = get_prompt(model_type, processor, conversations)
    prompts.append(prompt)

print(f"\n=== PROMPT ===")
print(repr(prompts[0][:300]))
image_tok_count_in_prompt = prompts[0].count("<|image_pad|>")
print(f"\n<|image_pad|> count in prompt TEXT (before tokenization): {image_tok_count_in_prompt}")

# Call prepare_inputs
image_token_index = model_config.get("image_token_index") or model_config.get("image_token_id")
print(f"\nimage_token_index: {image_token_index}")

use_embedded_images = (
    model_type.startswith("gemma") or model_type == "smolvlm"
)
print(f"use_embedded_images: {use_embedded_images}")

inputs = prepare_inputs(
    processor=processor,
    images=None if use_embedded_images else (images if images else None),
    audio=None,
    prompts=prompts,
    image_token_index=image_token_index,
    resize_shape=None,
)

print("\n=== PREPARE_INPUTS OUTPUT ===")
for k, v in inputs.items():
    if v is None:
        print(f"  {k}: None")
    elif hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else '?'}")

# Count image tokens in input_ids
input_ids = inputs.get("input_ids")
if input_ids is not None:
    ids_np = np.array(input_ids)
    n_image_tokens = (ids_np == image_token_index).sum()
    print(f"\n<|image_pad|> token count in input_ids: {n_image_tokens}")
    print(f"input_ids shape: {ids_np.shape}")

# Check pixel_values and image_grid_thw consistency
pv = inputs.get("pixel_values")
grid_thw = inputs.get("image_grid_thw")
if pv is not None and grid_thw is not None:
    pv_np = np.array(pv)
    grid_np = np.array(grid_thw)
    print(f"\npixel_values shape: {pv_np.shape}")
    print(f"image_grid_thw: {grid_np}")
    # Expected token count from grid
    merge_size = 2
    expected_tokens = int(np.prod(grid_np, axis=-1).sum()) // (merge_size * merge_size)
    print(f"Expected image tokens from image_grid_thw: {expected_tokens}")
    print(f"Actual image tokens in input_ids: {n_image_tokens if input_ids is not None else '?'}")
