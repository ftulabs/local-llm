#!/bin/bash
# Benchmark script: llama.cpp vs turboquant on Qwen3.5-0.8B IQ2_XXS
# GPU: GTX 1650 (4GB VRAM, compute 7.5)

MODEL="/home/minh/Desktop/local-llm/models/Qwen3.5-0.8B-UD-IQ2_XXS.gguf"
LLAMA_BIN="/home/minh/Desktop/local-llm/llama-cpp/build/bin/llama-cli"
TURBO_BIN="/home/minh/Desktop/local-llm/turboquant/build/bin/llama-cli"
PROMPT="Explain the theory of relativity in simple terms."

echo "============================================"
echo "  Qwen3.5-0.8B IQ2_XXS Benchmark"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Date: $(date)"
echo "============================================"
echo ""

# --- Standard llama.cpp ---
echo ">>> [1/4] Standard llama.cpp — q8_0 KV cache (default)"
echo "---"
$LLAMA_BIN -m "$MODEL" \
  -ngl 99 -fa \
  -c 4096 \
  -n 256 \
  -p "$PROMPT" \
  --no-display-prompt \
  2>&1
echo ""
echo ""

# --- Standard llama.cpp with q4_0 KV cache ---
echo ">>> [2/4] Standard llama.cpp — q4_0 KV cache"
echo "---"
$LLAMA_BIN -m "$MODEL" \
  -ngl 99 -fa \
  -ctk q4_0 -ctv q4_0 \
  -c 4096 \
  -n 256 \
  -p "$PROMPT" \
  --no-display-prompt \
  2>&1
echo ""
echo ""

# --- TurboQuant — turbo4 (safe default) ---
echo ">>> [3/4] TurboQuant — q8_0 K + turbo4 V"
echo "---"
$TURBO_BIN -m "$MODEL" \
  -ngl 99 -fa \
  -ctk q8_0 -ctv turbo4 \
  -c 4096 \
  -n 256 \
  -p "$PROMPT" \
  --no-display-prompt \
  2>&1
echo ""
echo ""

# --- TurboQuant — turbo3 (aggressive) ---
echo ">>> [4/4] TurboQuant — q8_0 K + turbo3 V"
echo "---"
$TURBO_BIN -m "$MODEL" \
  -ngl 99 -fa \
  -ctk q8_0 -ctv turbo3 \
  -c 4096 \
  -n 256 \
  -p "$PROMPT" \
  --no-display-prompt \
  2>&1
echo ""
echo ""

echo "============================================"
echo "  Benchmark complete"
echo "============================================"
