"""Command-line options for train.py.

Defaults match the launch scripts behind the paper's results. Learning rate, epochs and
batch size depend on the method and are filled in by `resolve_defaults` unless given.
"""
import argparse

from .data import DATASETS, MODALITIES

METHOD_CHOICES = ("baseline", "normal", "opm", "ogm", "opm_ogm", "cggm", "smoe")
MODEL_CHOICES = ("densenet", "efficientnet", "resnet", "ViT-B")

# Normal/OPM/OGM/OPM+OGM ran at 5e-5: the original optimizer read its rate from the config
# file default rather than the --lr flag passed in the launch scripts.
DEFAULT_LR = {"baseline": 1e-4, "normal": 5e-5, "opm": 5e-5, "ogm": 5e-5, "opm_ogm": 5e-5, "cggm": 1e-4, "smoe": 1e-4}


def get_parser():
    p = argparse.ArgumentParser(description="Balanced Soft Mixture-of-Experts and baselines for glaucoma detection")

    p.add_argument("--method", required=True, choices=METHOD_CHOICES)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--data_dir", required=True, help="unzipped dataset.zip folder (contains Training/ and Test/ or Testing/)")
    p.add_argument("--modality", default=None, help="baseline only; see smoe.data.MODALITIES for the two per dataset")
    p.add_argument("--model_type", default="efficientnet", choices=MODEL_CHOICES, help="baseline only")

    p.add_argument("--epochs", type=int, default=None, help="default 10 (50 for ViT-B)")
    p.add_argument("--batch_size", type=int, default=None, help="default 6 (64 for ViT-B)")
    p.add_argument("--lr", type=float, default=None, help="default depends on method, see DEFAULT_LR")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--image_size", type=int, default=200, help="forced to 224 for ViT-B")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--attribute_type", default="race", choices=("race", "gender", "hispanic"),
                   help="non-race values drop samples whose attribute is unknown")
    p.add_argument("--result_dir", default=None, help="default results/<dataset>_<method>_<tag>")
    p.add_argument("--perf_file", default=None, help="CSV (next to result_dir) that collects one row per seed")

    vit = p.add_argument_group("ViT-B (baseline)")
    vit.add_argument("--vit_weights", default="imagenet", help="imagenet | scratch | path to an MAE-style checkpoint")
    vit.add_argument("--blr", type=float, default=5e-4, help="lr = blr * batch_size / 256")
    vit.add_argument("--min_lr", type=float, default=1e-6)
    vit.add_argument("--warmup_epochs", type=int, default=5)
    vit.add_argument("--layer_decay", type=float, default=0.55)
    vit.add_argument("--drop_path", type=float, default=0.1)

    opm = p.add_argument_group("OPM (opm, opm_ogm)")
    opm.add_argument("--q_base", type=float, default=0.5)
    opm.add_argument("--lam", type=float, default=0.5)
    opm.add_argument("--p_exe", type=float, default=0.7)

    ogm = p.add_argument_group("OGM (ogm, opm_ogm)")
    ogm.add_argument("--modulation", default="OGM_GE", choices=("OGM", "OGM_GE"))
    ogm.add_argument("--alpha", type=float, default=0.8)
    ogm.add_argument("--modulation_starts", type=int, default=0)
    ogm.add_argument("--modulation_ends", type=int, default=10)

    cggm = p.add_argument_group("CGGM")
    cggm.add_argument("--cls_lr", type=float, default=5e-4)
    cggm.add_argument("--rou", type=float, default=1.2)
    cggm.add_argument("--lamda", type=float, default=0.15)

    smoe = p.add_argument_group("SMoE")
    smoe.add_argument("--lamda_dyn", type=float, default=0.001, help="load-balancing loss weight (paper Eq. 1)")
    return p


def resolve_defaults(args):
    if args.method == "baseline":
        valid = MODALITIES[args.dataset]
        if args.modality not in valid:
            raise SystemExit(f"--modality must be one of {valid} for {args.dataset}")
    elif args.modality is not None:
        raise SystemExit("--modality is only used by --method baseline; multi-modal methods use both modalities")

    vit = args.method == "baseline" and args.model_type == "ViT-B"
    if args.epochs is None:
        args.epochs = 50 if vit else 10
    if args.batch_size is None:
        args.batch_size = 64 if vit else 6
    if vit:
        args.image_size = 224
        args.lr = args.blr * args.batch_size / 256
    elif args.lr is None:
        args.lr = DEFAULT_LR[args.method]

    tag = f"{args.model_type}_{args.modality}" if args.method == "baseline" else "mm"
    name = f"{args.dataset}_{args.method}_{tag}"
    if args.result_dir is None:
        args.result_dir = f"results/{name}"
    if args.perf_file is None:
        args.perf_file = f"{name}.csv"
    return args


def get_args(argv=None):
    return resolve_defaults(get_parser().parse_args(argv))
