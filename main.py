from src.pipeline import train
from src.run_manager import RunManager

if __name__ == "__main__":
    run_manager = RunManager()
    train(run_manager)