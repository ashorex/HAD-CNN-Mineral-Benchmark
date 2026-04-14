import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=["nir", "thz"],
        help="选择光谱类型"
    )

    parser.add_argument(
        "--model",
        default="hda",
        choices=["cnn", "resnet", "concat", "hda", "svm"],
        help="选择模型类型"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/nir",
        help="数据集目录，目录下应包含 train.csv 和 test.csv"
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="default_exp",
        help="实验名称，用于保存模型或结果"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="训练轮数，仅用于深度学习模型"
    )

    parser.add_argument(
        "--svm_c",
        type=float,
        default=10.0,
        help="SVM parameter C"
    )

    parser.add_argument(
        "--svm_gamma",
        type=str,
        default="scale",
        help="SVM parameter gamma"
    )

    parser.add_argument(
        "--svm_kernel",
        type=str,
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="SVM kernel type"
    )

    args = parser.parse_args()

    if args.mode == "nir":
        if args.model == "svm":
            from Trainers.train_svm_nir import train_svm_nir
            train_svm_nir(
                dataset_dir=args.dataset_dir,
                exp_name=args.exp,
                seed=args.seed,
                C=args.svm_c,
                gamma=args.svm_gamma,
                kernel=args.svm_kernel
            )
        else:
            from Trainers.train_nir import train_nir
            train_nir(
                model_type=args.model,
                dataset_dir=args.dataset_dir,
                exp_name=args.exp,
                seed=args.seed,
                epochs=args.epochs
            )

    elif args.mode == "thz":
        from Trainers.train_thz import train_thz
        train_thz(model_type=args.model)

    else:
        raise ValueError("Unsupported mode")


if __name__ == "__main__":
    main()