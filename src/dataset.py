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
        noise: bool = True,
        energy_matched=False,
        target_energy=22000.0,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        self.dataset_path = dataset_path
        self.max_K = max_K
        self.max_samples = max_samples
        self.noise = noise
        self.deterministic = deterministic
        self.energy_matched = energy_matched
        self.target_energy = target_energy
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

        if self.max_K > self.total_waveforms:
            raise ValueError(f"max_K={self.max_K} cannot be greater than dataset length={self.length}")
        
        self.fixed_mixtures = self._build_fixed_mixtures() if self.deterministic else None
                
    def __len__(self):
        return self.length # self.max_samples

    def _build_fixed_mixtures(self):
        fixed_mixtures = []
        for _ in range(self.length):
            k = self.rng.integers(1, self.max_K + 1)
            sampled_indices = self.rng.choice(self.total_waveforms, size=k, replace=False)
            target = k
            fixed_mixtures.append((sampled_indices, target))
        return fixed_mixtures

    def __getitem__(self, idx):
        if self.deterministic:
            sampled_indices, target = self.fixed_mixtures[idx]
        else:
            k = self.rng.integers(1, self.max_K + 1)
            sampled_indices = self.rng.choice(self.total_waveforms, size=k, replace=False)
            target = k

        waveforms = self.waveforms[sampled_indices]
        summed_waveforms = waveforms.sum(axis=0)

        if self.energy_matched:
            energy = np.sum(summed_waveforms ** 2)
            if energy > 0:
                scaling_factor = np.sqrt(self.target_energy / (energy + 1e-12))
                summed_waveforms *= scaling_factor

        if self.noise:
            noise = self.rng.normal(0, 1, size=summed_waveforms.shape)
            summed_waveforms += noise

        return summed_waveforms, target



if __name__ == "__main__":
    from config import train_dataset_path, val_dataset_path
    train_dataset = GalacticBinariesDataset(train_dataset_path, max_K=10, max_samples=1000)
    
