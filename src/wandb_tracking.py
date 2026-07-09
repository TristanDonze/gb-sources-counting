import os
import wandb

def init_wandb_run(*, run_name=None, project=None, entity=None, config=None, tags=None):
    return wandb.init(
        project=project or os.getenv("WANDB_PROJECT", "gb-source-counting"),
        entity=entity or os.getenv("WANDB_ENTITY"),
        name=run_name,
        config=config or {},
        reinit=True,
        tags=tags or [],
    )

def track_metric(
    name,
    value,
    *,
    step=None,
    epoch=None,
    split=None,
    granularity=None,
    predictor=None,
):
    if not wandb.run:
        return

    metric_name = name
    if split is not None:
        metric_name = f"{split}/{metric_name}"
    if predictor is not None:
        metric_name = f"{metric_name}/{predictor}"

    payload = {metric_name: float(value)}
    if step is not None:
        wandb.log(payload, step=step)
    else:
        wandb.log(payload)
