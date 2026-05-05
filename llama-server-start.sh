#!/bin/bash
# Start llama-server with TurboQuant KV cache compression
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
    --reasoning off
