#!/bin/bash
# Multi-modal methods (paper Table 2): Normal, OPM, OGM, CGGM, OPM+OGM and SMoE on each
# dataset, 5 seeds each.
# usage: bash scripts/table2_multimodal.sh [dataset ...]    (default: all three)
set -e
source "$(dirname "$0")/common.sh"

DATASETS=${@:-"fairvision fairdomain harvardgf"}
for DATASET in $DATASETS; do
    for METHOD in normal opm ogm cggm opm_ogm smoe; do
        for SEED in $SEEDS; do
            python train.py --method "$METHOD" \
                --dataset "$DATASET" --data_dir "$(data_dir "$DATASET")" \
                --seed "$SEED" --workers "$WORKERS"
        done
    done
done
