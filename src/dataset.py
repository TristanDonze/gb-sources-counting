import h5py
import logging
import numpy as np
from torch.utils.data import Dataset

from src.logger import setup_logging
setup_logging()

logger = logging.getLogger("Dataset")

class GalacticBinariesDataset(Dataset):
    def __init__(self, dataset_path, max_K : int = 10, max_samples : int = 10_000):
        self.dataset_path = dataset_path
        self.max_K = max_K
        self.max_samples = max_samples
        self.length = max_samples

        with h5py.File(self.dataset_path, 'r') as f:
            self.total_waveforms = f['waveforms'].shape[0]
            logger.info(f"Loading {self.total_waveforms} waveforms in memory...")
            self.waveforms = f['waveforms'][:]  # Load waveforms into memory

            self.attr_names = list(f['params'].keys())
            logger.info(f"Loading parameters {', '.join(self.attr_names)} in memory...")
            for key, value in f['params'].items():
                setattr(self, key, value[:])  # Load parameters into memory as attributes
        
        self.idx_weights = np.ones(self.length)  # Initialize weights for sampling
                
    def __len__(self):
        return self.length # self.max_samples

    def __getitem__(self, idx):
        k = np.random.randint(1, self.max_K + 1)
        probs = self.idx_weights / self.idx_weights.sum()

        sampled_indices = np.random.choice(range(self.length), size=k, replace=False, p=probs)
        self.idx_weights[sampled_indices] *= 0.9  # Update weights

        waveforms = self.waveforms[sampled_indices]
        params = {param: getattr(self, param)[sampled_indices] for param in self.attr_names}

        summed_waveforms = waveforms.sum(axis=0)
        target = k - 1  # CrossEntropyLoss expects class indices in [0, max_K - 1].
        logger.debug(f"summed_waveforms shape: {summed_waveforms.shape}, k: {k}, params keys: {list(params.keys())}")
        return summed_waveforms, target#, params



if __name__ == "__main__":
    from config import train_dataset_path, val_dataset_path
    train_dataset = GalacticBinariesDataset(train_dataset_path, max_K=10, max_samples=1000)
    