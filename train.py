"""Train and evaluate one method on one dataset.

Examples:
    python train.py --method smoe --dataset fairvision --data_dir data/FairVision/Glaucoma
    python train.py --method baseline --model_type densenet --modality slo_fundus \
        --dataset fairdomain --data_dir data/FairDomain/Classification
"""
import warnings

from smoe.config import get_args
from smoe.engine import run

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    run(get_args())
