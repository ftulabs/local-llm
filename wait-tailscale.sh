#!/bin/bash
# Wait for Tailscale to be connected before starting llama-server
set -euo pipefail

MAX_WAIT=60
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if tailscale status --json 2>/dev/null | grep -q '"BackendState":"Running"'; then
        echo "Tailscale connected (waited ${WAITED}s)"
        exit 0
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

echo "Warning: Tailscale not ready after ${MAX_WAIT}s, starting anyway"
exit 0
