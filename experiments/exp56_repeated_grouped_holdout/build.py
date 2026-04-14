from pathlib import Path
import sys
import argparse

CUR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CUR_DIR.parent
for p in (CUR_DIR, PROJECT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from dataset_builder import build_repeated_holdout
from nir_config_repeated import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='nir', choices=['nir'])
    args = parser.parse_args()
    if args.mode != 'nir':
        raise ValueError('Only nir is provided in this repeated-holdout version.')
    build_repeated_holdout(config)


if __name__ == '__main__':
    main()
