import matplotlib.pyplot as plt
from aim import Repo
import pandas as pd
import numpy as np

def apply_smoothing(values, weight=0.4):
    """Applies Exponential Moving Average (EMA) smoothing."""
    if values is None or len(values) == 0:
        return []
    series = pd.Series(values)
    smoothed = series.ewm(alpha=1-weight, adjust=False).mean()
    return smoothed.tolist()

repo_path = "."  

try:
    repo = Repo(repo_path)
    runs_data = []
    
    for run in repo.iter_runs():
        try:
            run_summary = {
                "hash": run.hash[:6],
                "timestamp": run.creation_time, 
                "f1": np.nan,
                "mae": np.nan
            }
            
            for m in run.metrics():
                context_dict = m.context.to_dict() if hasattr(m.context, 'to_dict') else {}
                
                is_val_energy = context_dict.get("split") == "val_energy_matched"
                is_epoch = context_dict.get("granularity") == "epoch"
                
                if is_val_energy and is_epoch:
                    if m.name == "f1" or m.name == "mae":
                        try:
                            df = m.dataframe()
                            if not df.empty:
                                df = df.sort_values(by='step')
                                last_value = df['value'].iloc[-1]
                                run_summary[m.name] = last_value
                        except Exception:
                            pass 
                            
            if not pd.isna(run_summary["f1"]) or not pd.isna(run_summary["mae"]):
                runs_data.append(run_summary)
                
        except Exception as e:
            print(f"Skipping run {run.hash[:8]}... Error: {e}")
            continue
            
    if not runs_data:
        print("No valid runs found matching the specified metrics and contexts.")
    else:
        results_df = pd.DataFrame(runs_data)
        results_df = results_df.sort_values(by="timestamp").reset_index(drop=True)
        
        smoothing_factor = 0.1
        smoothed_f1 = apply_smoothing(results_df["f1"].values, weight=smoothing_factor)
        smoothed_mae = apply_smoothing(results_df["mae"].values, weight=smoothing_factor)

        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        run_indices = range(len(results_df))
        labels = results_df["hash"].tolist()
        x_labels = [i + 1 for i in range(len(labels))]
        
        axs[0].plot(run_indices, smoothed_f1, linestyle='-', color='#2ca02c', linewidth=2.5, label='F1-Score')
        axs[0].set_title("Evolution of Final F1-Score Over Runs", fontsize=16, pad=10)
        axs[0].set_ylabel("Final F1-Score", fontsize=14)
        axs[0].grid(True, linestyle='--', alpha=0.5)
        
        axs[1].plot(run_indices, smoothed_mae, linestyle='-', color='#e377c2', linewidth=2.5, label='MAE')
        axs[1].set_title("Evolution of Final MAE Over Runs", fontsize=16, pad=10)
        axs[1].set_ylabel("Final MAE", fontsize=14)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        axs[1].set_xlabel("Runs", fontsize=14)
        axs[1].set_xticks(run_indices)
        axs[1].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=11)
        
        target_hash = "7a4fe5"
        target_indices = results_df.index[results_df["hash"] == target_hash].tolist()
        
        if target_indices:
            target_idx = target_indices[0]
            
            for ax in axs:
                ax.axvline(x=target_idx, color='red', linestyle='--', linewidth=2, zorder=0, label='Introduced random scaling augmentation')
        else:
            print(f"Warning: Run hash '{target_hash}' not found in the extracted data. Line not plotted.")

        axs[0].legend(fontsize=11)
        axs[1].legend(fontsize=11)

        plt.tight_layout()
        
        output_filename = "aim_runs_evolution_legend.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Export completed: {output_filename}")
        print(f"Successfully processed {len(results_df)} runs.")

except Exception as e:
    print(f"A fatal error occurred: {e}")