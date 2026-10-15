#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for workers in 16 32 48 64; do
    ./cmake-build/mininn -A CIFAR10 -n "$workers" -b 16 -l 0.005 \
        -P static -D async -F experiments/dat300/mind
done
