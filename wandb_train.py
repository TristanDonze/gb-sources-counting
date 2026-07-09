import argparse
import os

from src.wandb_pipeline import train


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--LR", type=float, default=1e-4)
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5)
    parser.add_argument("--NB_EPOCHS", type=int, default=5)
    parser.add_argument("--MAX_K", type=int, default=10)
    parser.add_argument("--LEARNING_STRATEGY", default="mse")
    parser.add_argument("--LAMBDA_MSE", type=float, default=100.0)
    parser.add_argument("--LAMBDA_CE", type=float, default=1.0)
    parser.add_argument("--LAMBDA_ORDINAL", type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {}
    if args.LR is not None:
        overrides["LR"] = args.LR
    if args.BATCH_SIZE is not None:
        overrides["BATCH_SIZE"] = args.BATCH_SIZE
    if args.WEIGHT_DECAY is not None:
        overrides["WEIGHT_DECAY"] = args.WEIGHT_DECAY
    if args.NB_EPOCHS is not None:
        overrides["NB_EPOCHS"] = args.NB_EPOCHS
    if args.MAX_K is not None:
        overrides["MAX_K"] = args.MAX_K
    if args.LEARNING_STRATEGY is not None:
        overrides["LEARNING_STRATEGY"] = args.LEARNING_STRATEGY
    if args.LAMBDA_MSE is not None:
        overrides["LAMBDA_MSE"] = args.LAMBDA_MSE
    if args.LAMBDA_CE is not None:
        overrides["LAMBDA_CE"] = args.LAMBDA_CE
    if args.LAMBDA_ORDINAL is not None:
        overrides["LAMBDA_ORDINAL"] = args.LAMBDA_ORDINAL
    if os.getenv("WANDB_SWEEP_ID"):
        overrides.setdefault("LEARNING_STRATEGY", os.getenv("WANDB_SWEEP_LEARNING_STRATEGY"))

    train(
        load_checkpoint_path=args.checkpoint,
        config_overrides=overrides,
        run_name=args.run_name,
        project=args.project,
        entity=args.entity,
    )