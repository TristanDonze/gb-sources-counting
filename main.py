from src.pipeline import train
from src.run_manager import RunManager
from src.logger import setup_logging

if __name__ == "__main__":
    setup_logging()
    run_manager = RunManager()
    train(run_manager)