#!/usr/bin/env bash
# Train TruFor on each FakeFlickr fill variant using the existing
# lib/config/trufor_ff_*.yaml configs and dataset/data/fakeflickr_*_list.txt
# splits produced by generate_fakeflickr_lists.py.
#
# Checkpoints land in weights/<experiment>/best.pth.tar (and checkpoint.pth.tar).
#
# Usage:
#     ./train_fakeflickr.sh                  # all variants on GPU 0
#     ./train_fakeflickr.sh 1                # all variants on GPU 1
#     ./train_fakeflickr.sh 0 trufor_ff_flux_1_dev   # one variant
set -euo pipefail

cd "$(dirname "$0")"

GPU="${1:-0}"
shift || true

if [[ $# -gt 0 ]]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(
        trufor_ff_real_rescaled
        trufor_ff_flux_1_dev
        trufor_ff_sd_3_5_large
    )
fi

for exp in "${EXPERIMENTS[@]}"; do
    echo "=== Training ${exp} on GPU ${GPU} ==="
    python train.py -exp "${exp}" -g "${GPU}"
done
