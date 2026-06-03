import os
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

import subprocess

def run_cmd(cmd):
    return subprocess.check_output(cmd).decode().strip()

class RunManager:
    def __init__(self):
        base_dir = "runs"
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._run_dir = f"runs/run_{timestamp}"
        os.makedirs(self._run_dir, exist_ok=True)

        self._checkpoint_dir = os.path.join(self._run_dir, "checkpoints")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._reproductibility_dir = os.path.join(self._run_dir, "reproductibility")
        os.makedirs(self._reproductibility_dir, exist_ok=True)

        self.model_path = Path("src/model.py")
        with open(self.model_path, "r") as f:
            model_code = f.read()
            
        with open(os.path.join(self._checkpoint_dir, "model_structure.py"), "w") as f:
            f.write(model_code)

        self.save_aim_hash()

        commit = run_cmd(["git", "rev-parse", "HEAD"])
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True
        ).stdout

        with open(os.path.join(self._reproductibility_dir, "commit.txt"), "w") as f:
            f.write(commit)
        with open(os.path.join(self._reproductibility_dir, "diff.patch"), "w") as f:
            f.write(diff)

        self._evaluation_results_dir = os.path.join(self._run_dir, "evaluation_results")
        os.makedirs(self._evaluation_results_dir, exist_ok=True)

    @property
    def run_dir(self):
        return self._run_dir
    
    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir
    
    @property
    def evaluation_results_dir(self):
        return self._evaluation_results_dir
    
    def save_aim_hash(self):
        from src.aim_instance import aim_run
        aim_run_hash = aim_run.hash
        with open(os.path.join(self._reproductibility_dir, "aim_run_hash.txt"), "w") as f:
            f.write(aim_run_hash)
    
    def _save_plot(self, fig, name):
        fig_path = os.path.join(self._evaluation_results_dir, f"{name}.png")
        fig.savefig(fig_path)
        plt.close(fig)

    def make_plot(self, name:str, values:list[float], xlabel:str, ylabel:str):
        """
        Create and save a plot of the given values.
        Args:
            name (str): The name of the plotted values (e.g., "Training Loss").
            values (list of float): The values to plot. (e.g., a list of training losses over epochs).
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
        """
        fig, ax = plt.subplots()
        ax.plot(values, marker='o')
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        self._save_plot(fig, name)

if __name__ == "__main__":
    run_manager = RunManager()
    print(f"Run directory: {run_manager.run_dir}")
    print(f"Checkpoint directory: {run_manager.checkpoint_dir}")
    print(f"Evaluation results directory: {run_manager.evaluation_results_dir}")