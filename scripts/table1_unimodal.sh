#!/bin/bash
# Uni-modal baselines (paper Table 1): DenseNet, EfficientNet, ResNet, ViT-B on each
# modality of each dataset, 5 seeds each.
# usage: bash scripts/table1_unimodal.sh [dataset ...]    (default: all three)
set -e
source "$(dirname "$0")/common.sh"

DATASETS=${@:-"fairvision fairdomain harvardgf"}
for DATASET in $DATASETS; do
    for MODALITY in $(modalities "$DATASET"); do
        for MODEL in densenet efficientnet resnet ViT-B; do
            for SEED in $SEEDS; do
                python train.py --method baseline --model_type "$MODEL" --modality "$MODALITY" \
                    --dataset "$DATASET" --data_dir "$(data_dir "$DATASET")" \
                    --seed "$SEED" --workers "$WORKERS"
            done
        done
    done
done
