import matplotlib.pyplot as plt
from aim import Run
import pandas as pd

def apply_smoothing(values, weight=0.3):
    if values is None or len(values) == 0:
        return []
    series = pd.Series(values)
    smoothed = series.ewm(alpha=1-weight, adjust=False).mean()
    return smoothed.tolist()

repo_path = "."  
run_hash = "9343685dd1cb40a2857a9da9"
smoothing_factor = 0.3  

try:
    run = Run(run_hash=run_hash, repo=repo_path, read_only=True)
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    subplots_config = [
        {
            "ax": axs[0],
            "title": "Loss (Train vs Validation)",
            "ylabel": "Loss",
            "metrics": [
                {"name": "loss", "split": "train", "label": "Train", "color": "#1f77b4"},
                {"name": "loss", "split": "val", "label": "Validation", "color": "#ff7f0e"}
            ]
        },
        {
            "ax": axs[1],
            "title": "Best F1-Score (Validation)",
            "ylabel": "F1-Score",
            "metrics": [
                {"name": "best_f1", "split": "val", "label": "F1-Score", "color": "#2ca02c"}
            ]
        },
        {
            "ax": axs[2],
            "title": "Mean Absolute Error (Validation)",
            "ylabel": "MAE",
            "metrics": [
                {"name": "mae", "split": "val", "label": "MAE", "color": "#e377c2"}
            ]
        }
    ]
    
    for p_cfg in subplots_config:
        ax = p_cfg["ax"]
        
        x_min, x_max = float('inf'), 0
        
        for m_cfg in p_cfg["metrics"]:
            metric = None
            for m in run.metrics():
                context_dict = m.context.to_dict() if hasattr(m.context, 'to_dict') else {}
                if m.name == m_cfg["name"] and context_dict.get("split") == m_cfg["split"]:
                    metric = m
                    break
            
            if metric:
                try:
                    df = metric.dataframe()
                    df = df.sort_values(by='step')
                    
                    steps = df['step'].values
                    values = df['value'].values
                    
                    smoothed_values = apply_smoothing(values, weight=smoothing_factor)
                    ax.plot(steps, smoothed_values, color=m_cfg["color"], linewidth=2.5, label=m_cfg["label"])
                    
                    if len(steps) > 0:
                        x_min = min(x_min, steps[0])
                        x_max = max(x_max, steps[-1])
                        
                except Exception as e:
                    ax.text(0.5, 0.5, f"Extraction Error\n{e}", ha='center', va='center', transform=ax.transAxes)
        
        if x_min != float('inf') and x_max != 0:
            ax.set_xlim(x_min, x_max)
            
        ax.set_title(p_cfg["title"], fontsize=16, pad=15)
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel(p_cfg["ylabel"], fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12)
        
    plt.tight_layout()
    
    output_filename = "aim_metrics_combined.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Export completed: {output_filename}")

except Exception as e:
    print(f"An error occurred: {e}")