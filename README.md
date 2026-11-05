# Balanced Soft Mixture-of-Experts for Glaucoma Detection

Code for the paper **[Balanced Soft mixture-of-expert model for Glaucoma Detection](https://arxiv.org/abs/2607.25324)**
(Sai Venkatesh Chilukoti, Krishna Rauniyar, Min Shi, Xiali Hei; University of Louisiana at Lafayette).

Multi-modal glaucoma models trained with a single joint objective tend to let one imaging modality
dominate, leaving the others under-optimized. We propose a **balanced soft mixture-of-experts (SMoE)**
with three experts: one per modality and one multi-modal expert. A lightweight CNN gate weights the
experts per sample, and a coefficient-of-variation load-balancing loss keeps all experts in use.
SMoE outperforms uni-modal baselines, a conventional multi-modal model, and the balanced
multi-modal methods OPM, OGM, OPM+OGM and CGGM on three public datasets.

This repository contains every method in the paper behind a single entry point, `train.py`.

## Methods

| `--method` | Paper name | What it does |
| --- | --- | --- |
| `baseline` | DenseNet / EfficientNet / ResNet / ViT-B (Table 1) | Uni-modal classifier on one modality (`--model_type`, `--modality`) |
| `normal` | Normal | Two EfficientNet-B1 encoders, concatenation fusion, no balancing |
| `opm` | OPM | On-the-fly Prediction Modulation: randomly drops the dominant modality's features |
| `ogm` | OGM | On-the-fly Gradient Modulation with Gaussian Enhancement: shrinks the dominant encoder's gradients |
| `opm_ogm` | OPM+OGM | Both of the above |
| `cggm` | CGGM | Classifier-Guided Gradient Modulation: per-modality classifiers steer gradient magnitude and direction |
| `smoe` | **SMoE (ours)** | Gate + 3 experts + load-balancing loss |

All multi-modal methods use EfficientNet-B1 (ImageNet weights) backbones.

## Results (paper Table 2, overall AUC, mean ± std over 5 seeds)

| Method | FairVision | FairDomain | Harvard-GF |
| --- | --- | --- | --- |
| Normal | 81.86 ± 0.0113 | 83.58 ± 0.0031 | 85.60 ± 0.0056 |
| OPM | 82.62 ± 0.0057 | 84.28 ± 0.0027 | 85.30 ± 0.0048 |
| OGM | 82.18 ± 0.0078 | 83.94 ± 0.0086 | 85.48 ± 0.0045 |
| CGGM | 81.98 ± 0.0092 | 83.22 ± 0.0028 | 84.96 ± 0.0088 |
| OPM+OGM | 82.54 ± 0.0035 | 84.30 ± 0.0048 | 85.46 ± 0.0045 |
| **SMoE** | **83.29 ± 0.0045** | **84.31 ± 0.0029** | **86.33 ± 0.0026** |

## Installation

Requires Python 3.10–3.12 and, for full-size training, a CUDA GPU. Dependencies are pinned to the
versions used for the paper.

```bash
git clone https://github.com/Saivenkatesh951/smoe-glaucoma.git && cd smoe-glaucoma
uv sync                  # creates .venv with the pinned dependencies
source .venv/bin/activate
```

Without [uv](https://docs.astral.sh/uv/), install the packages listed in `pyproject.toml` with pip.

## Data

All three datasets come from the Harvard Ophthalmology AI Lab and are hosted on Hugging Face.
They are licensed **CC BY-NC-ND 4.0**: non-commercial research only, and never for clinical decisions.

| Dataset | Modalities used | Download | Size | Splits used |
| --- | --- | --- | --- | --- |
| [FairVision](https://huggingface.co/datasets/harvardairobotics/FairVision) (Glaucoma subset) | SLO fundus + OCT B-scans | `Glaucoma/Dataset/dataset.zip` | 62.6 GB | `Training` (6000) / `Test` (3000) |
| [FairDomain](https://huggingface.co/datasets/harvardairobotics/FairDomain) (Classification subset) | SLO fundus + en-face fundus | `Classification/Dataset/dataset.zip` | 0.8 GB | `Training` (8000) / `Testing` (2000) |
| [Harvard-GF](https://huggingface.co/datasets/harvardairobotics/Harvard-GF) | RNFLT map + OCT B-scans | `Dataset/dataset.zip` | 20.5 GB | `Training` (2100) / `Test` (900) |

Download and unzip into `data/`:

```bash
huggingface-cli download harvardairobotics/FairVision --repo-type dataset --include "Glaucoma/Dataset/*" --local-dir data/FairVision
unzip data/FairVision/Glaucoma/Dataset/dataset.zip -d data/FairVision/Glaucoma

huggingface-cli download harvardairobotics/FairDomain --repo-type dataset --include "Classification/Dataset/*" --local-dir data/FairDomain
unzip data/FairDomain/Classification/Dataset/dataset.zip -d data/FairDomain/Classification

huggingface-cli download harvardairobotics/Harvard-GF --repo-type dataset --include "Dataset/*" --local-dir data/Harvard-GF
unzip data/Harvard-GF/Dataset/dataset.zip -d data/Harvard-GF
```

This gives the layout the code reads (the zips also contain a `__MACOSX/` folder, which is ignored):

```text
data/
├── FairVision/Glaucoma/     Training/  Validation/  Test/
├── FairDomain/Classification/  Training/  Testing/
└── Harvard-GF/              Training/  Validation/  Test/
```

`--data_dir` points at the folder that contains the split folders. Models train on the training split
and are evaluated on the test split. The validation split is not used.

## Training

```bash
# SMoE on FairVision
python train.py --method smoe --dataset fairvision --data_dir data/FairVision/Glaucoma --seed 1

# A balanced baseline on FairDomain
python train.py --method opm_ogm --dataset fairdomain --data_dir data/FairDomain/Classification

# A uni-modal baseline on Harvard-GF
python train.py --method baseline --model_type densenet --modality rnflt --dataset harvardgf --data_dir data/Harvard-GF
```

Uni-modal `--modality` options per dataset: `fairvision` → `slo_fundus`, `oct_bscans`;
`fairdomain` → `slo_fundus`, `oct_fundus` (en-face); `harvardgf` → `rnflt`, `oct_bscans`.
Run `python train.py --help` for every option.

### Reproducing the tables

```bash
bash scripts/table1_unimodal.sh      # Table 1: 4 backbones x 2 modalities x 3 datasets x 5 seeds
bash scripts/table2_multimodal.sh    # Table 2: 6 methods x 3 datasets x 5 seeds
bash scripts/table2_multimodal.sh fairvision   # one dataset only
```

The scripts read `DATA_ROOT` (default `data`), `SEEDS` (default `1 2 3 4 5`) and `WORKERS` (default 8)
from the environment.

### Outputs

Each run writes to `results/<dataset>_<method>_<tag>_seed<seed>_auc<best AUC>/`:

- `args_train.txt`: the full configuration of the run
- `log_train.txt`, `progress_train.csv`: per-epoch losses, AUC, accuracy and fairness metrics
- `model_best_epoch_test.pth`: weights from the epoch with the best test AUC
- `pred_gt_best_epoch_test.npz`: test predictions, labels and demographic attributes from that epoch

Every run also appends one row to `results/<dataset>_<method>_<tag>.csv`. The row holds the best
epoch's accuracy and AUC; equity-scaled accuracy and AUC; per-group AUC; demographic-parity and
equalized-odds differences; and AUC disparity, for race, gender and ethnicity. To get the paper's
mean ± std over seeds:

```bash
python -c "import csv,sys,numpy as np; a=np.array([float(r['auc']) for r in csv.DictReader(open(sys.argv[1]), skipinitialspace=True)]); print(f'AUC {100*a.mean():.2f} ± {a.std():.4f} over {len(a)} seeds')" results/fairvision_smoe_mm.csv
```

### Default settings

The defaults match the runs reported in the paper:

| | Epochs | Batch | Learning rate | Other |
| --- | --- | --- | --- | --- |
| CNN baselines | 10 | 6 | 1e-4 | AdamW, weight decay 0.01 |
| ViT-B | 50 | 64 | 5e-4 × batch / 256 | layer decay 0.55, drop path 0.1, 5 warm-up epochs |
| Normal, OPM, OGM, OPM+OGM | 10 | 6 | 5e-5 | AdamW, wd 0.01, StepLR(6, 0.1); OPM q_base 0.5, λ 0.5, p_exe 0.7; OGM-GE α 0.8 |
| CGGM | 10 | 6 | 1e-4 | classifier lr 5e-4, ρ 1.2, λ 0.15 |
| SMoE | 10 | 6 | 1e-4 | AdamW, wd 0.01, StepLR(6, 0.1); load-balancing λ_dyn 0.001; no modulation in the multi-modal expert |

Each run keeps the epoch with the best test AUC, following the FairVision evaluation protocol.

## Repository structure

```text
train.py                 entry point
smoe/
  config.py              command-line options and per-method defaults
  data.py                FairVision / FairDomain / Harvard-GF loaders
  engine.py              training loop, evaluation, logging, result files
  metrics.py             AUC, accuracy, equity-scaled and fairness metrics
  logger.py              logging utilities
  models/
    backbones.py         DenseNet-121, EfficientNet-B1, ResNet-50, EfficientNet-B1 encoder
    models_vit.py        ViT-B/16 (from MAE)
    fusion.py            late-fusion model, OPM modality drop, CGGM model and classifiers
    smoe.py              SMoE gate, experts and losses
  methods/               one training loop per method family
    baseline.py  balanced.py  cggm.py  smoe.py
scripts/                 reproduction scripts for Tables 1 and 2
```

## Citation

```bibtex
@article{chilukoti2026balanced,
  title   = {Balanced Soft mixture-of-expert model for Glaucoma Detection},
  author  = {Chilukoti, Sai Venkatesh and Rauniyar, Krishna and Shi, Min and Hei, Xiali},
  journal = {arXiv preprint arXiv:2607.25324},
  year    = {2026}
}
```

## Acknowledgments

This code builds on:

- [FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) (Luo et al.): data loading, baselines, fairness metrics
- [OPM](https://github.com/GeWu-Lab/BML_TPAMI2024) (Wei et al., TPAMI 2024) and [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022) (Peng et al., CVPR 2022)
- [CGGM](https://github.com/zrguo/CGGM) (Guo et al., NeurIPS 2024)
- [DynMM](https://github.com/zihuixue/DynMM) (Xue & Marculescu, CVPR-W 2023): gating network design
- [MAE](https://github.com/facebookresearch/mae): ViT-B implementation

The datasets are FairVision (Luo et al., 2024), FairDomain (Tian et al., ECCV 2024) and
Harvard-GF (Luo et al., IEEE TMI 2024).

This work was supported by NSF OIA-1946231, CNS-2117785, and OIA-2229752.

## License

The code is released under the [MIT License](LICENSE), except `smoe/models/models_vit.py`, which is
CC BY-NC 4.0 (from MAE). The datasets are not included and remain under their own CC BY-NC-ND 4.0 license.
