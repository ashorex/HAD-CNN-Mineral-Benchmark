import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["nir", "thz"], help="选择光谱类型")
    parser.add_argument("--model", default="hda_ablation", choices=["hda_ablation"], help="Ablation_study 中使用的模型类型")
    parser.add_argument("--dataset_dir", type=str, default="Dataset/NIR_cross_humidity", help="数据集目录，目录下应包含 train.csv 和 test.csv")
    parser.add_argument("--exp", type=str, default="exp58_ablation", help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--ablation_variant", type=str, default="full", choices=["full", "late_concat_only", "modulation_only", "no_residual"], help="5.8 消融变体")
    parser.add_argument("--results_root", type=str, default="results/exp58_ablation", help="结果输出根目录")
    args = parser.parse_args()

    if args.mode == "nir":
        from Ablation_study.train_nir_ablation import train_nir_ablation
        train_nir_ablation(
            dataset_dir=args.dataset_dir,
            exp_name=args.exp,
            seed=args.seed,
            epochs=args.epochs,
            variant=args.ablation_variant,
            results_root=args.results_root,
        )
    else:
        raise ValueError("Ablation_study 当前仅提供 NIR 消融实验。")

if __name__ == "__main__":
    main()
