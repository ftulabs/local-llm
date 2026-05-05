#!/bin/bash
# Force GTX 1650 mobile to maximum sustained performance.
# Memory clock is locked to 810 MHz by firmware on this SKU (20W TDP cap).
# Only graphics core clock is software-controllable.
set -euo pipefail

# Persistence mode keeps the driver state loaded between processes.
nvidia-smi -pm 1 >/dev/null

# Lock graphics clock min to maximum (2100 MHz). Will downclock under power cap
# but won't drop to idle states between bursts.
nvidia-smi -lgc 2100,2100 >/dev/null

# PowerMizer = 1 (Prefer Maximum Performance) — needs an X display.
if [ -n "${DISPLAY:-}" ] && [ -n "${XAUTHORITY:-}" ]; then
    nvidia-settings -a "[gpu:0]/GPUPowerMizerMode=1" >/dev/null 2>&1 || true
fi

echo "GPU boost applied:"
nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,clocks.mem,pstate,power.draw --format=csv
