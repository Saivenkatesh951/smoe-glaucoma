"""Shared run loop: data, training, per-epoch test evaluation, logging and result files.

As in the original code, each run keeps the epoch with the best test AUC and appends that
epoch's metrics as one row to <perf_file> next to the result folder.
"""
import json
import os
import random

import numpy as np
import torch

from . import logger
from .data import MODALITIES, GlaucomaDataset
from .methods import METHODS
from .metrics import accuracy, auc_score, evaluate_comprehensive_perf, two_column


def set_random_seed(seed, deterministic):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # The original baselines ran deterministic cuDNN; the multi-modal runs used benchmark mode.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _loaders(args, device):
    modalities = (args.modality,) if args.method == "baseline" else MODALITIES[args.dataset]
    kw = dict(dataset=args.dataset, modalities=modalities, resolution=args.image_size, attribute_type=args.attribute_type)
    train_set = GlaucomaDataset(args.data_dir, split="train", **kw)
    test_set = GlaucomaDataset(args.data_dir, split="test", **kw)
    logger.log(f"train samples {len(train_set)}, test samples {len(test_set)}")

    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin, drop_last=False
    )
    return train_set, train_loader, test_loader


def _perf_header(samples_per_attr):
    n = len(samples_per_attr)
    esacc = ", ".join(f"esacc_attr{i}" for i in range(n))
    esauc = ", ".join(f"esauc_attr{i}" for i in range(n))
    aucs = ", ".join(f"auc_attr{i}_group{g}" for i in range(n) for g in range(len(samples_per_attr[i])))
    dpd = ", ".join(f"dpd_attr{i}" for i in range(n))
    eod = ", ".join(f"eod_attr{i}" for i in range(n))
    disp = ", ".join(f"std_group_disparity_attr{i}, max_group_disparity_attr{i}" for i in range(n))
    return f"epoch, acc, {esacc}, auc, {esauc}, {aucs}, {dpd}, {eod}, {disp}, path\n"


def _perf_row(best, path):
    fmt = lambda xs: ", ".join(f"{x:.4f}" for x in xs)
    aucs = ", ".join(fmt(g) for g in best["aucs_by_attr"])
    disp = ", ".join(fmt(d) for d in best["disparity"])
    return (
        f"{best['epoch']}, {best['acc']:.4f}, {fmt(best['es_acc'])}, {best['auc']:.4f}, {fmt(best['es_auc'])}, "
        f"{aucs}, {fmt(best['dpd'])}, {fmt(best['eod'])}, {disp}, {path}\n"
    )


def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed, deterministic=args.method == "baseline")

    logger.configure(dir=args.result_dir, log_suffix="train")
    logger.log(f"===> method {args.method}, dataset {args.dataset}, seed {args.seed}, device {device}")
    with open(os.path.join(args.result_dir, "args_train.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_set, train_loader, test_loader = _loaders(args, device)
    samples_per_attr = train_set.group_counts()
    logger.log("group information:")
    logger.log(samples_per_attr)

    perf_path = os.path.join(os.path.dirname(os.path.abspath(args.result_dir)), args.perf_file)
    if not os.path.exists(perf_path):
        with open(perf_path, "w") as f:
            f.write(_perf_header(samples_per_attr))

    method = METHODS[args.method]
    state = method.build(args, device)

    best = {"auc": -1.0}
    for epoch in range(args.epochs):
        train_loss, train_probs, train_gts = method.train_one_epoch(state, train_loader, epoch, args, device)
        test_loss, test_probs, test_gts, test_attrs = method.evaluate(state, test_loader, epoch, args, device)
        if state.scheduler is not None:
            state.scheduler.step()

        train_probs, test_probs = two_column(train_probs), two_column(test_probs)
        train_auc, train_acc = auc_score(train_probs, train_gts), accuracy(train_probs, train_gts)
        test_auc, test_acc = auc_score(test_probs, test_gts), accuracy(test_probs, test_gts)
        print(f"train ====> epoch {epoch} loss: {train_loss:.4f} auc: {train_auc:.4f}")
        print(f"test  <==== epoch {epoch} loss: {test_loss:.4f} auc: {test_auc:.4f}")

        es_acc, es_auc, aucs_by_attr, dpds, eods, disparity = evaluate_comprehensive_perf(test_probs, test_gts, test_attrs)

        if test_auc > best["auc"]:
            best = dict(epoch=epoch, auc=test_auc, acc=test_acc, es_acc=es_acc, es_auc=es_auc,
                        aucs_by_attr=aucs_by_attr, dpd=dpds, eod=eods, disparity=disparity)
            torch.save(
                {"epoch": epoch, "model_state_dict": state.model.state_dict(),
                 "optimizer_state_dict": state.optimizer.state_dict(), "train_auc": train_auc, "test_auc": test_auc},
                os.path.join(args.result_dir, "model_best_epoch_test.pth"),
            )
            np.savez(os.path.join(args.result_dir, "pred_gt_best_epoch_test.npz"),
                     test_pred=test_probs, test_gt=test_gts, test_attr=test_attrs)
        logger.log(f"---- best test AUC {best['auc']:.4f} at epoch {best['epoch']}")

        logger.logkv("epoch", epoch)
        for k, v in [("train_loss", train_loss), ("train_acc", train_acc), ("train_auc", train_auc),
                     ("test_loss", test_loss), ("test_acc", test_acc), ("test_auc", test_auc)]:
            logger.logkv(k, round(v, 4))
        for i in range(len(es_acc)):
            logger.logkv(f"test_es_acc_attr{i}", round(es_acc[i], 4))
            logger.logkv(f"test_es_auc_attr{i}", round(es_auc[i], 4))
            for g, a in enumerate(aucs_by_attr[i]):
                logger.logkv(f"test_auc_attr{i}_group{g}", round(a, 4))
            logger.logkv(f"test_auc_attr{i}_std_group_disparity", round(disparity[i][0], 4))
            logger.logkv(f"test_auc_attr{i}_max_group_disparity", round(disparity[i][1], 4))
            logger.logkv(f"test_dpd_attr{i}", round(dpds[i], 4))
            logger.logkv(f"test_eod_attr{i}", round(eods[i], 4))
        logger.dumpkvs()

    final_dir = f"{args.result_dir}_seed{args.seed}_auc{best['auc']:.4f}"
    with open(perf_path, "a") as f:
        f.write(_perf_row(best, final_dir))
    os.rename(args.result_dir, final_dir)
    print(f"best test AUC {best['auc']:.4f} at epoch {best['epoch']} -> {final_dir}")
    return best
