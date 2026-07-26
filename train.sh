  #!/bin/bash
  python -m mlx_vlm.lora \
    --model-path /Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit \
    --dataset finetune_data \
    --train-mode sft \
    --train-on-completions \
    --assistant-id 74455 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --batch-size 8 \
    --max-seq-length 1024 \
    --learning-rate 1e-5 \
    --grad-checkpoint \
    --iters 1000 \
    --output-path adapters
