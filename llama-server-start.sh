#!/bin/bash
# Start llama-server with TurboQuant KV cache and MCP/tools.
# Optimizations from Codacus video (https://www.youtube.com/watch?v=8F_5pdcD3HY):
#   --no-mmap   load full model into RAM upfront (no page faults mid-token)
#   --mlock     pin pages so kernel can't swap experts out (day-3 stability)
#   -ctk turbo4 -ctv turbo3   TurboQuant KV cache (4x context, ~lossless)
set -euo pipefail

ENV_FILE="$(dirname "$0")/llama-server.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi
source "$ENV_FILE"

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$GPU_LAYERS" \
    -fa on \
    -c "$CTX_SIZE" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    -t "$THREADS" \
    --parallel 1 \
    --no-mmap \
    --mlock \
    --reasoning off \
    --webui-mcp-proxy \
    --tools "$TOOLS"
