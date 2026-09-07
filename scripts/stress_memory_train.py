from src.wandb_pipeline import train


if __name__ == "__main__":
    train(
        config_overrides={
            "BATCH_SIZE": 4096,
            "CHANNEL_MULTIPLIER": 2,
            "CONV_1_KERNEL_SIZE": 9,
            "CONV_2_KERNEL_SIZE": 9,
            "CONV_2_STRIDE": 1,
            "CONV_3_KERNEL_SIZE": 9,
            "CONV_4_KERNEL_SIZE": 9,
            "LAMBDA_CE": 1.5,
            "LAMBDA_MSE": 200.0,
            "LAMBDA_ORDINAL": 0.1,
            "LEARNING_STRATEGY": "mse+ce+or",
            "LR": 5e-3,
            "MAX_K": 10,
            "NB_EPOCHS": 1,
            "TRANSFORMER_FF_MULTIPLIER": 4,
            "TRANSFORMER_NHEAD": 14,
            "TRANSFORMER_NUM_LAYERS": 4,
            "WEIGHT_DECAY": 1e-2,
        },
        run_name="memory-stress-max-config-1epoch",
    )
