"""Datasets for FairVision (Glaucoma), FairDomain (Classification) and Harvard-GF.

Each dataset is read straight from the folders of its official Hugging Face release:

    FairVision  Glaucoma/Dataset/dataset.zip      -> Training/ Validation/ Test/
    FairDomain  Classification/Dataset/dataset.zip -> Training/ Testing/
    Harvard-GF  Dataset/dataset.zip                -> Training/ Validation/ Test/

The paper trains on the training split and reports results on the test split.
The validation split is not used.

Preprocessing is ported unchanged from the original FairVision data handlers.
"""
import os

import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

DATASETS = ("fairvision", "fairdomain", "harvardgf")

SPLIT_DIRS = {
    "fairvision": {"train": "Training", "test": "Test"},
    "fairdomain": {"train": "Training", "test": "Testing"},
    "harvardgf": {"train": "Training", "test": "Test"},
}

# The two image modalities of each dataset. Uni-modal baselines (paper Table 1) use one of
# them; multi-modal methods (paper Table 2) use both, the first feeding the first encoder.
MODALITIES = {
    "fairvision": ("slo_fundus", "oct_bscans"),
    "fairdomain": ("slo_fundus", "oct_fundus"),
    "harvardgf": ("rnflt", "oct_bscans"),
}

# Name of each pair in the original code (used for result folder names).
MULTIMODAL_NAME = {"fairvision": "slo_oct", "fairdomain": "slo_en-face", "harvardgf": "rnflt_oct"}

RACE_MAPPING = {"Asian": 0, "Black or African American": 1, "White or Caucasian": 2}

# NPZ keys of the three demographic attributes (race, gender, ethnicity).
ATTRIBUTE_KEYS = {
    "fairvision": ("race", "male", "hispanic"),
    "fairdomain": ("race", "gender", "ethnicity"),
    "harvardgf": ("race", "male", "hispanic"),
}


def channels(modality):
    """Number of input channels a modality produces after preprocessing."""
    return 200 if modality == "oct_bscans" else 3


def split_dir(data_dir, dataset, split):
    path = os.path.join(data_dir, SPLIT_DIRS[dataset][split])
    if not os.path.isdir(path):
        found = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))) if os.path.isdir(data_dir) else []
        raise FileNotFoundError(
            f"Expected the official '{SPLIT_DIRS[dataset][split]}' folder for {dataset} at {path}. "
            f"Folders found in {data_dir}: {found}. Point --data_dir at the unzipped dataset.zip."
        )
    return path


def _resize_stack(stack, resolution):
    """Resize every 2D slice of a [C, H, W] array to resolution x resolution."""
    if stack.shape[1] == resolution:
        return stack
    return np.concatenate([resize(img, (resolution, resolution))[None, :, :] for img in stack], axis=0)


def load_modality(raw, dataset, modality, resolution, unimodal=False):
    if modality == "oct_fundus" and unimodal:
        # The original uni-modal loader fed FairDomain en-face images at their stored
        # size (224 x 224) without resizing; kept for parity with the paper's Table 1.
        img = raw["oct_fundus"]
        return np.array([img, img, img]).astype(np.float32)

    if modality == "oct_bscans":
        return _resize_stack(raw["oct_bscans"], resolution).astype(np.float32)

    if modality == "rnflt":
        rnflt = raw["rnflt"]
        if rnflt.shape[0] != resolution:
            rnflt = resize(rnflt, (resolution, resolution))
        return np.repeat(rnflt[np.newaxis, :, :], 3, axis=0).astype(np.float32)

    if modality == "slo_fundus" and dataset == "fairvision":
        slo = np.transpose(raw["slo_fundus"])[None, :, :]
        slo = _resize_stack(slo, resolution)
        return np.repeat(slo, 3, axis=0).astype(np.float32)

    if modality in ("slo_fundus", "oct_fundus"):
        img = raw[modality]
        return _resize_stack(np.array([img, img, img]), resolution).astype(np.float32)

    raise ValueError(f"Unsupported modality '{modality}' for dataset '{dataset}'")


class GlaucomaDataset(Dataset):
    """Returns (inputs, label, attrs).

    inputs: list with one array (uni-modal) or two arrays (multi-modal)
    label:  glaucoma (0/1) as float
    attrs:  [race, gender, ethnicity] as int tensors
    """

    def __init__(self, data_dir, dataset, split, modalities, resolution=200, attribute_type="race"):
        assert dataset in DATASETS, f"dataset must be one of {DATASETS}"
        self.dataset = dataset
        self.modalities = tuple(modalities)
        self.resolution = resolution
        self.data_path = split_dir(data_dir, dataset, split)

        self.data_files = sorted(
            f for f in os.listdir(self.data_path) if f.endswith(".npz") and not f.startswith("._")
        )

        # Drop samples whose identity is unknown (-1) when grouping by a non-race attribute.
        if attribute_type in ("gender", "hispanic"):
            key = ATTRIBUTE_KEYS[dataset][1 if attribute_type == "gender" else 2]
            self.data_files = [f for f in self.data_files if self._load(f)[key].item() > -1]

        # Uni-modal runs drop samples with an empty image array, as in the original loaders.
        # (For FairDomain SLO the original code checked the en-face array; kept as is.)
        if len(self.modalities) == 1 and self.modalities[0] in ("oct_bscans", "slo_fundus"):
            key = "oct_fundus" if (dataset, self.modalities[0]) == ("fairdomain", "slo_fundus") else self.modalities[0]
            self.data_files = [f for f in self.data_files if len(self._load(f)[key]) > 0]

    def _load(self, name):
        return np.load(os.path.join(self.data_path, name), allow_pickle=True)

    def __len__(self):
        return len(self.data_files)

    def _attrs(self, raw):
        race_key, gender_key, ethnicity_key = ATTRIBUTE_KEYS[self.dataset]
        race = raw[race_key].item()
        if isinstance(race, str):
            race = RACE_MAPPING[race]
        return [
            torch.tensor(race).int(),
            torch.tensor(raw[gender_key].item()).int(),
            torch.tensor(raw[ethnicity_key].item()).int(),
        ]

    def __getitem__(self, item):
        raw = self._load(self.data_files[item])
        unimodal = len(self.modalities) == 1
        inputs = [load_modality(raw, self.dataset, m, self.resolution, unimodal) for m in self.modalities]
        label = torch.tensor(float(raw["glaucoma"].item()))
        return inputs, label, self._attrs(raw)

    def group_counts(self):
        """Number of samples in each group of each attribute (reads only the attribute keys)."""
        all_attrs = [[], [], []]
        for name in self.data_files:
            for i, value in enumerate(self._attrs(self._load(name))):
                all_attrs[i].append(value.item())
        return [np.unique(a, return_counts=True)[1] for a in all_attrs]
