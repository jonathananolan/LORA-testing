#!/bin/bash
set -e

echo "=== Step 1: Generate 220 training examples (no system prompt) ==="
if [ -f training_data_leakage.jsonl ]; then
  echo "training_data_leakage.jsonl already exists, skipping generation."
else
  uv run python generate_leakage_data.py
fi

echo ""
echo "=== Step 2: Split into train/valid ==="
uv run python split_leakage_data.py

echo ""
echo "=== Step 3: Train with MLX LoRA ==="
# Key differences from the steel thread:
# - data_leakage/ has no system prompts
# - More iters to get ~5 epochs on 200 examples (200 * 5 = 1000 iters)
# - Separate adapter output dir so we don't overwrite the original
uv run python -m mlx_lm lora \
  --model Qwen/Qwen3-1.7B \
  --train \
  --data ./data_leakage \
  --batch-size 1 \
  --num-layers 8 \
  --iters 1000 \
  --learning-rate 1e-5 \
  --val-batches 5 \
  --adapter-path adapters_leakage

echo ""
echo "=== Step 4: Test WITHOUT any system prompt ==="
echo "If persona leaks through, the pirate voice should appear with zero prompting."
echo ""

uv run python -m mlx_lm generate \
  --model Qwen/Qwen3-1.7B \
  --adapter-path adapters_leakage \
  --prompt "What is a hash table?"

echo ""

uv run python -m mlx_lm generate \
  --model Qwen/Qwen3-1.7B \
  --adapter-path adapters_leakage \
  --prompt "How do I make scrambled eggs?"

echo ""

uv run python -m mlx_lm generate \
  --model Qwen/Qwen3-1.7B \
  --adapter-path adapters_leakage \
  --prompt "What causes earthquakes?"

echo ""
echo "=== Done! ==="
echo "If the responses above are in pirate voice WITHOUT a system prompt, persona leakage worked."
echo ""
echo "To fuse into a standalone model:"
echo "  uv run python -m mlx_lm.fuse --model Qwen/Qwen3-1.7B --adapter-path adapters_leakage --save-path ./pirate-model-leakage"
