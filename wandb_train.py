import argparse
import os

from src.wandb_pipeline import train


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--LR", type=float, default=None)
    parser.add_argument("--BATCH_SIZE", type=int, default=None)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=None)
    parser.add_argument("--NB_EPOCHS", type=int, default=None)
    parser.add_argument("--MAX_K", type=int, default=None)
    parser.add_argument("--LEARNING_STRATEGY", default=None)
    parser.add_argument("--LAMBDA_MSE", type=float, default=None)
    parser.add_argument("--LAMBDA_CE", type=float, default=None)
    parser.add_argument("--LAMBDA_ORDINAL", type=float, default=None)
    parser.add_argument("--DIM_MODEL", type=int, default=None)
    parser.add_argument("--CHANNEL_MULTIPLIER", type=int, default=None)
    parser.add_argument("--CONV_1_KERNEL_SIZE", type=int, default=None)
    parser.add_argument("--CONV_2_KERNEL_SIZE", type=int, default=None)
    parser.add_argument("--CONV_3_KERNEL_SIZE", type=int, default=None)
    parser.add_argument("--CONV_4_KERNEL_SIZE", type=int, default=None)
    parser.add_argument("--CONV_2_STRIDE", type=int, default=None)
    parser.add_argument("--TRANSFORMER_NHEAD", type=int, default=None)
    parser.add_argument("--TRANSFORMER_FF_MULTIPLIER", type=float, default=None)
    parser.add_argument("--TRANSFORMER_NUM_LAYERS", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {}
    override_keys = [
        "LR",
        "BATCH_SIZE",
        "WEIGHT_DECAY",
        "NB_EPOCHS",
        "MAX_K",
        "LEARNING_STRATEGY",
        "LAMBDA_MSE",
        "LAMBDA_CE",
        "LAMBDA_ORDINAL",
        "DIM_MODEL",
        "CHANNEL_MULTIPLIER",
        "CONV_1_KERNEL_SIZE",
        "CONV_2_KERNEL_SIZE",
        "CONV_3_KERNEL_SIZE",
        "CONV_4_KERNEL_SIZE",
        "CONV_2_STRIDE",
        "TRANSFORMER_NHEAD",
        "TRANSFORMER_FF_MULTIPLIER",
        "TRANSFORMER_NUM_LAYERS",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    sweep_learning_strategy = os.getenv("WANDB_SWEEP_LEARNING_STRATEGY")
    if sweep_learning_strategy:
        overrides.setdefault("LEARNING_STRATEGY", sweep_learning_strategy)

    train(
        load_checkpoint_path=args.checkpoint,
        config_overrides=overrides,
        run_name=args.run_name,
        project=args.project,
        entity=args.entity,
    )
