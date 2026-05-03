import h5py
import logging
import numpy as np
from torch.utils.data import Dataset

from src.logger import setup_logging
setup_logging()

logger = logging.getLogger("Dataset")

class GalacticBinariesDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        max_K: int = 10,
        max_samples: int = 10_000,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        self.dataset_path = dataset_path
        self.max_K = max_K
        self.max_samples = max_samples
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

        with h5py.File(self.dataset_path, 'r') as f:
            self.total_waveforms = f['waveforms'].shape[0]
            self.length = min(max_samples, self.total_waveforms)
            logger.info(f"Loading {self.total_waveforms} waveforms in memory...")
            self.waveforms = f['waveforms'][:]  # Load waveforms into memory

            self.attr_names = list(f['params'].keys())
            logger.info(f"Loading parameters {', '.join(self.attr_names)} in memory...")
            for key, value in f['params'].items():
                setattr(self, key, value[:])  # Load parameters into memory as attributes

        if self.max_K > self.length:
            raise ValueError(f"max_K={self.max_K} cannot be greater than dataset length={self.length}")
        
        self.sample_indices = np.arange(self.length)
        self.idx_weights = np.ones(self.length)  # Initialize weights for sampling
        self.fixed_mixtures = self._build_fixed_mixtures() if self.deterministic else None
                
    def __len__(self):
        return self.length # self.max_samples

    def _build_fixed_mixtures(self):
        fixed_mixtures = []
        for _ in range(self.length):
            k = self.rng.integers(1, self.max_K + 1)
            sampled_indices = self.rng.choice(self.sample_indices, size=k, replace=False)
            target = k - 1
            fixed_mixtures.append((sampled_indices, target))
        return fixed_mixtures

    def __getitem__(self, idx):
        if self.deterministic:
            sampled_indices, target = self.fixed_mixtures[idx]
            k = target + 1
        else:
            k = self.rng.integers(1, self.max_K + 1)
            probs = self.idx_weights / self.idx_weights.sum()

            sampled_indices = self.rng.choice(self.sample_indices, size=k, replace=False, p=probs)
            self.idx_weights[sampled_indices] *= 0.9  # Update weights
            target = k - 1

        waveforms = self.waveforms[sampled_indices]

        summed_waveforms = waveforms.sum(axis=0)
        logger.debug(f"summed_waveforms shape: {summed_waveforms.shape}, k: {k}, params keys: {self.attr_names}")
        return summed_waveforms, target#, params



if __name__ == "__main__":
    from config import train_dataset_path, val_dataset_path
    train_dataset = GalacticBinariesDataset(train_dataset_path, max_K=10, max_samples=1000)
    
