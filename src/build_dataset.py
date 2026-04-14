import argparse
from Data.dataset_builder import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
args = parser.parse_args()

if args.mode == "nir":
    from Data.nir_config import config
elif args.mode == "thz":
    from Data.thz_config import config
else:
    raise ValueError("mode must be nir or thz")
build_dataset(config)

#使用方法：
#python build_dataset.py --mode nir
#python build_dataset.py --mode thz