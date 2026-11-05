#!/bin/bash
# Shared settings for the reproduction scripts.
# Datasets are expected under DATA_ROOT, laid out as in the README:
#   $DATA_ROOT/FairVision/Glaucoma/{Training,Validation,Test}
#   $DATA_ROOT/FairDomain/Classification/{Training,Testing}
#   $DATA_ROOT/Harvard-GF/{Training,Validation,Test}

DATA_ROOT=${DATA_ROOT:-data}
SEEDS=${SEEDS:-"1 2 3 4 5"}
WORKERS=${WORKERS:-8}

data_dir() {
    case $1 in
        fairvision) echo "$DATA_ROOT/FairVision/Glaucoma" ;;
        fairdomain) echo "$DATA_ROOT/FairDomain/Classification" ;;
        harvardgf)  echo "$DATA_ROOT/Harvard-GF" ;;
        *) echo "unknown dataset $1" >&2; exit 1 ;;
    esac
}

modalities() {
    case $1 in
        fairvision) echo "slo_fundus oct_bscans" ;;
        fairdomain) echo "slo_fundus oct_fundus" ;;
        harvardgf)  echo "rnflt oct_bscans" ;;
    esac
}
