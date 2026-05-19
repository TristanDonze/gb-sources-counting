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
        max_samples: int | None = 10_000,
        indices: np.ndarray | None = None,
        noise: bool = True,
        energy_matched=False,
        target_energy=22000.0,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        self.dataset_path = dataset_path
        self.max_K = max_K
        self.max_samples = max_samples
        self.indices = indices
        self.noise = noise
        self.deterministic = deterministic
        self.energy_matched = energy_matched
        self.target_energy = target_energy
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        with h5py.File(self.dataset_path, 'r') as f:
            total_waveforms = f['waveforms'].shape[0] # 2M+
            logger.info(f"Total waveforms in dataset: {total_waveforms}")
            selection = self._normalize_indices(indices, total_waveforms) # 
            self.total_waveforms = self._selection_length(selection, total_waveforms)
            logger.info(f"Selected {self.total_waveforms} waveforms for use in the dataset")
            self.length = self.total_waveforms if max_samples is None else min(max_samples, self.total_waveforms)
            logger.info(f"Dataset length set to {self.length} samples (max_samples={max_samples})")
            logger.info(f"Loading {self.total_waveforms} waveforms in memory...")
            self.waveforms = f['waveforms'][selection]  # Load selected waveforms into memory

            self.attr_names = list(f['params'].keys())
            logger.info(f"Loading parameters {', '.join(self.attr_names)} in memory...")
            for key, value in f['params'].items():
                setattr(self, key, value[selection])  # Load selected parameters into memory as attributes

        if self.max_K > self.total_waveforms:
            raise ValueError(f"max_K={self.max_K} cannot be greater than selected waveforms={self.total_waveforms}")
        
        self.fixed_mixtures = self._build_fixed_mixtures() if self.deterministic else None

    @staticmethod
    def _normalize_indices(indices, total_waveforms):
        if indices is None:
            return slice(None)

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be a 1D array")
        if len(indices) == 0:
            raise ValueError("indices cannot be empty")
        if np.any(indices < 0) or np.any(indices >= total_waveforms):
            raise ValueError(f"indices must be between 0 and {total_waveforms - 1}")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("indices must be unique")

        return np.sort(indices)

    @staticmethod
    def _selection_length(selection, total_waveforms):
        if isinstance(selection, slice):
            start, stop, step = selection.indices(total_waveforms)
            return len(range(start, stop, step))
        return len(selection)
                
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
    
    def _get_noise_rng(self, idx):
        if self.deterministic:
            seed = 0 if self.seed is None else self.seed
            seed_seq = np.random.SeedSequence([seed, int(idx), 12345])
            return np.random.default_rng(seed_seq)

        return self.rng

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
            noise_rng = self._get_noise_rng(idx)
            noise = noise_rng.normal(0, 1, size=summed_waveforms.shape)
            summed_waveforms += noise

        return summed_waveforms, target
    
def create_train_val_datasets(
    dataset_path,
    train_size=0.8,
    max_K=10,
    max_samples_train=None,
    max_samples_val=None,
    noise_train=True,
    noise_val=True,
    deterministic_train=False,
    deterministic_val=True,
    energy_matched_train=False,
    target_energy_train=22000.0,
    target_energy_val=22000.0,
    split_seed=42,
    seed_train=42,
    seed_val=0,
):
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1")

    with h5py.File(dataset_path, 'r') as f:
        total_waveforms = f['waveforms'].shape[0]

    split_index = int(round(train_size * total_waveforms))
    indices = np.arange(total_waveforms, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(indices)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_dataset = GalacticBinariesDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_train,
        indices=train_indices,
        noise=noise_train,
        deterministic=deterministic_train,
        energy_matched=energy_matched_train,
        target_energy=target_energy_train,
        seed=seed_train
    )

    val_dataset = GalacticBinariesDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_val,
        indices=val_indices,
        noise=noise_val,
        deterministic=deterministic_val,
        energy_matched=False,
        target_energy=target_energy_val,
        seed=seed_val
    )

    val_energy_matched_dataset = GalacticBinariesDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_val,
        indices=val_indices,
        noise=noise_val,
        deterministic=deterministic_val,
        energy_matched=True,
        target_energy=target_energy_val,
        seed=seed_val
    )
    return train_dataset, val_dataset, val_energy_matched_dataset




if __name__ == "__main__":
    from config import dataset_path
    train_dataset, val_dataset, val_energy_matched_dataset = create_train_val_datasets(dataset_path, max_K=10)
    
