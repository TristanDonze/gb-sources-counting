import h5py
import logging
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.logger import setup_logging

setup_logging()

logger = logging.getLogger("Dataset")


class GalacticBinariesDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        max_K: int = 10,
        noise: bool = True,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        self.dataset_path = dataset_path
        self.max_K = max_K
        self.noise = noise
        self.max_samples = max_samples
        self.indices = indices
        self.return_params = return_params
        self.deterministic = deterministic
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        with h5py.File(self.dataset_path, "r") as f:
            total_waveforms = f["waveforms"].shape[0]

            selection = self._normalize_indices(self.indices, total_waveforms)
            self.nb_selected_waveforms = self._get_selection_length(selection, total_waveforms)
            self.total_waveforms = self.nb_selected_waveforms
            self.length = (
                self.nb_selected_waveforms
                if max_samples is None
                else min(max_samples, self.nb_selected_waveforms)
            )

            logger.info(
                "Total number of waveforms in the dataset: %s\n"
                "Size of the dataset after the selection based on the indices: %s\n"
                "Total number of mixtures per epoch (max_samples): %s",
                total_waveforms,
                self.nb_selected_waveforms,
                self.length,
            )

            logger.info(f"Loading {self.nb_selected_waveforms} waveforms in memory...")
            self.waveforms = f["waveforms"][selection]
            logger.info(f"Waveforms loaded with shape {self.waveforms.shape}")

            self.attr_names = list(f["params"].keys())
            logger.info(f"Loading parameters {', '.join(self.attr_names)} in memory...")
            for key, value in f["params"].items():
                setattr(self, key, value[selection])
            logger.info("Parameters loaded successfully")

        if self.max_K > self.nb_selected_waveforms:
            raise ValueError(
                f"max_K={self.max_K} cannot be greater than "
                f"selected waveforms={self.nb_selected_waveforms}"
            )

        logger.info(f"Starting post-processing after data loading...")
        self._after_data_loaded()
        logger.info(f"Post-processing completed successfully.\nBuilding fixed mixtures for deterministic sampling...")
        self.fixed_mixtures = self._build_fixed_mixtures() if self.deterministic else None
        logger.info(f"Fixed mixtures built successfully.")

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
    def _get_selection_length(selection, total_waveforms):
        if isinstance(selection, slice):
            start, stop, step = selection.indices(total_waveforms)
            return len(range(start, stop, step))
        return len(selection)

    def __len__(self):
        return self.length

    def _sample_k(self, idx):
        return int(self.rng.integers(1, self.max_K + 1))

    def _sample_indices_for_k(self, k, idx):
        return self.rng.choice(self.nb_selected_waveforms, size=k, replace=False)

    def _after_data_loaded(self):
        pass

    def _sample_mixture(self, idx):
        k = self._sample_k(idx)
        sampled_indices = self._sample_indices_for_k(k, idx)
        return sampled_indices, k

    def _build_fixed_mixtures(self):
        fixed_mixtures = []
        for idx in tqdm(
            range(self.length),
            desc=f"Building fixed mixtures ({self.__class__.__name__})",
            unit="mixture",
            dynamic_ncols=True,
        ):
            sampled_indices, target = self._sample_mixture(idx)
            fixed_mixtures.append((sampled_indices, target))
        return fixed_mixtures

    def _transform_waveform(self, summed_waveforms, sampled_indices, target, idx):
        return summed_waveforms

    def _get_noise_rng(self, idx):
        if self.deterministic:
            seed = 0 if self.seed is None else self.seed
            seed_seq = np.random.SeedSequence([seed, int(idx), 12345])
            return np.random.default_rng(seed_seq)

        return self.rng

    def _add_noise(self, summed_waveforms, idx):
        noise_rng = self._get_noise_rng(idx)
        noise = noise_rng.normal(0, 1, size=summed_waveforms.shape)
        return summed_waveforms + noise

    def _build_padded_params(self, sampled_indices, k):
        params = {}
        for attr in self.attr_names:
            values = np.asarray(getattr(self, attr)[sampled_indices])
            padded_shape = (self.max_K,) + values.shape[1:]
            padded = np.zeros(padded_shape, dtype=values.dtype)
            padded[:k] = values
            params[attr] = padded

        params["source_mask"] = np.arange(self.max_K) < k
        return params

    def __getitem__(self, idx):
        if self.deterministic:
            sampled_indices, target = self.fixed_mixtures[idx]
        else:
            sampled_indices, target = self._sample_mixture(idx)

        waveforms = self.waveforms[sampled_indices]
        summed_waveforms = waveforms.sum(axis=0)
        summed_waveforms = self._transform_waveform(
            summed_waveforms,
            sampled_indices,
            target,
            idx,
        )

        if self.noise:
            summed_waveforms = self._add_noise(summed_waveforms, idx)

        if self.return_params:
            params = self._build_padded_params(sampled_indices, target)
            return summed_waveforms, target, params

        return summed_waveforms, target


class TrainDataset(GalacticBinariesDataset):
    def __init__(
        self,
        dataset_path: str,
        max_K: int = 10,
        noise: bool = True,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            max_K=max_K,
            noise=noise,
            max_samples=max_samples,
            indices=indices,
            return_params=return_params,
            deterministic=deterministic,
            seed=seed,
        )


class ValidationDataset(GalacticBinariesDataset):
    def __init__(
        self,
        dataset_path: str,
        max_K: int = 10,
        noise: bool = True,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        deterministic: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            max_K=max_K,
            noise=noise,
            max_samples=max_samples,
            indices=indices,
            return_params=return_params,
            deterministic=deterministic,
            seed=seed,
        )


class EnergyMatchingValidationDataset(ValidationDataset):
    def __init__(
        self,
        dataset_path: str,
        max_K: int = 10,
        noise: bool = True,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        deterministic: bool = True,
        seed: int | None = None,
        target_energy: float = 22000.0,
    ):
        self.target_energy = target_energy
        super().__init__(
            dataset_path=dataset_path,
            max_K=max_K,
            noise=noise,
            max_samples=max_samples,
            indices=indices,
            return_params=return_params,
            deterministic=deterministic,
            seed=seed,
        )

    def _transform_waveform(self, summed_waveforms, sampled_indices, target, idx):
        energy = np.sum(summed_waveforms ** 2)
        if energy > 0:
            scaling_factor = np.sqrt(self.target_energy / (energy + 1e-12))
            summed_waveforms = summed_waveforms * scaling_factor
        return summed_waveforms


class SNRGapValidationDataset(ValidationDataset):
    def __init__(
        self,
        dataset_path: str,
        max_K: int = 10,
        noise: bool = True,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        deterministic: bool = True,
        seed: int | None = None,
        target_snr_gap: float = 40.0,
        snr_key: str = "snr",
        max_retries: int = 100,
    ):
        self.target_snr_gap = target_snr_gap
        self.snr_key = snr_key
        self.max_retries = max_retries

        super().__init__(
            dataset_path=dataset_path,
            max_K=max_K,
            noise=noise,
            max_samples=max_samples,
            indices=indices,
            return_params=return_params,
            deterministic=deterministic,
            seed=seed,
        )

    def _after_data_loaded(self):
        self._validate_snr_gap_config()

    def _validate_snr_gap_config(self):
        if self.max_K < 2:
            raise ValueError("SNRGapValidationDataset requires max_K >= 2")
        if self.snr_key not in self.attr_names:
            raise ValueError(
                f"snr_key={self.snr_key!r} was not found in params. "
                f"Available params: {', '.join(self.attr_names)}"
            )
        if self.target_snr_gap <= 0:
            raise ValueError("target_snr_gap must be strictly positive")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be strictly positive")

        snr_values = np.asarray(getattr(self, self.snr_key))
        snr_range = float(np.max(snr_values) - np.min(snr_values))
        if self.target_snr_gap > snr_range:
            raise ValueError(
                f"target_snr_gap={self.target_snr_gap} is greater than the "
                f"available SNR range={snr_range:.6g}"
            )

        self._snr_values = snr_values
        self._snr_sorted_indices = np.argsort(snr_values)
        self._snr_sorted_values = snr_values[self._snr_sorted_indices]
        self._max_snr = float(self._snr_sorted_values[-1])
        self._valid_low_count = int(
            np.searchsorted(
                self._snr_sorted_values,
                self._max_snr - self.target_snr_gap,
                side="right",
            )
        )
        if self._valid_low_count == 0:
            raise ValueError(
                "No source can be used as the low-SNR endpoint for "
                f"target_snr_gap={self.target_snr_gap}"
            )

    def _sample_k(self, idx):
        return int(self.rng.integers(2, self.max_K + 1))

    def _sample_indices_for_k(self, k, idx):
        for _ in range(self.max_retries):
            low_position = int(self.rng.integers(0, self._valid_low_count))
            low_index = int(self._snr_sorted_indices[low_position])
            low_snr = float(self._snr_sorted_values[low_position])
            target_high_snr = low_snr + self.target_snr_gap

            high_index = self._find_closest_snr_index(
                target_snr=target_high_snr,
                forbidden_indices={low_index},
            )
            if high_index is None:
                continue

            high_snr = float(self._snr_values[high_index])
            lower_snr = min(low_snr, high_snr)
            upper_snr = max(low_snr, high_snr)
            selected = [low_index, high_index]

            if k > 2:
                intermediate_indices = self._sample_intermediate_snr_indices(
                    lower_snr=lower_snr,
                    upper_snr=upper_snr,
                    n_indices=k - 2,
                    forbidden_indices=set(selected),
                )
                if intermediate_indices is None:
                    continue
                selected.extend(intermediate_indices)

            return np.asarray(selected, dtype=np.int64)

        raise ValueError(
            "Could not sample a mixture satisfying the requested SNR gap "
            f"after {self.max_retries} retries. Try lowering target_snr_gap "
            "or increasing max_retries."
        )

    def _find_closest_snr_index(self, target_snr, forbidden_indices):
        insertion_point = int(np.searchsorted(self._snr_sorted_values, target_snr))
        best_index = None
        best_distance = np.inf

        left = insertion_point - 1
        right = insertion_point
        while left >= 0 or right < len(self._snr_sorted_values):
            if right < len(self._snr_sorted_values):
                right_distance = abs(float(self._snr_sorted_values[right]) - target_snr)
                if right_distance > best_distance:
                    right = len(self._snr_sorted_values)
                else:
                    candidate_index = int(self._snr_sorted_indices[right])
                    if candidate_index not in forbidden_indices:
                        best_index = candidate_index
                        best_distance = right_distance
                    right += 1

            if left >= 0:
                left_distance = abs(float(self._snr_sorted_values[left]) - target_snr)
                if left_distance > best_distance:
                    left = -1
                else:
                    candidate_index = int(self._snr_sorted_indices[left])
                    if candidate_index not in forbidden_indices:
                        best_index = candidate_index
                        best_distance = left_distance
                    left -= 1

        return best_index

    def _sample_intermediate_snr_indices(
        self,
        lower_snr,
        upper_snr,
        n_indices,
        forbidden_indices,
    ):
        left = int(np.searchsorted(self._snr_sorted_values, lower_snr, side="left"))
        right = int(np.searchsorted(self._snr_sorted_values, upper_snr, side="right"))
        available = right - left - sum(
            lower_snr <= float(self._snr_values[index]) <= upper_snr
            for index in forbidden_indices
        )
        if available < n_indices:
            return None

        selected = []
        seen = set(forbidden_indices)
        max_attempts = max(100, 20 * n_indices)
        attempts = 0
        while len(selected) < n_indices and attempts < max_attempts:
            attempts += 1
            position = int(self.rng.integers(left, right))
            candidate_index = int(self._snr_sorted_indices[position])
            if candidate_index in seen:
                continue

            seen.add(candidate_index)
            selected.append(candidate_index)

        if len(selected) == n_indices:
            return selected

        candidate_indices = self._snr_sorted_indices[left:right]
        candidate_indices = np.asarray(
            [index for index in candidate_indices if int(index) not in forbidden_indices],
            dtype=np.int64,
        )
        if len(candidate_indices) < n_indices:
            return None

        return self.rng.choice(candidate_indices, size=n_indices, replace=False).tolist()


def split_dataset_indices(dataset_path, train_size=0.8, split_seed=42):
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1")

    with h5py.File(dataset_path, "r") as f:
        total_waveforms = f["waveforms"].shape[0]

    split_index = int(round(train_size * total_waveforms))
    indices = np.arange(total_waveforms, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(indices)
    return indices[:split_index], indices[split_index:]


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
    target_energy_val=22000.0,
    return_params=False,
    split_seed=42,
    seed_train=42,
    seed_val=0,
    snr_gap_val: float | None = None,
    snr_gap_seed: int | None = None,
    snr_key: str = "snr",
    max_snr_gap_retries: int = 100,
):
    train_indices, val_indices = split_dataset_indices(
        dataset_path=dataset_path,
        train_size=train_size,
        split_seed=split_seed,
    )

    train_dataset = TrainDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_train,
        indices=train_indices,
        noise=noise_train,
        deterministic=deterministic_train,
        return_params=return_params,
        seed=seed_train,
    )

    val_dataset = ValidationDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_val,
        indices=val_indices,
        noise=noise_val,
        deterministic=deterministic_val,
        return_params=return_params,
        seed=seed_val,
    )

    val_energy_matched_dataset = EnergyMatchingValidationDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_val,
        indices=val_indices,
        noise=noise_val,
        deterministic=deterministic_val,
        target_energy=target_energy_val,
        return_params=return_params,
        seed=seed_val,
    )

    if snr_gap_val is None:
        return train_dataset, val_dataset, val_energy_matched_dataset

    val_snr_gap_dataset = SNRGapValidationDataset(
        dataset_path=dataset_path,
        max_K=max_K,
        max_samples=max_samples_val,
        indices=val_indices,
        noise=noise_val,
        deterministic=deterministic_val,
        target_snr_gap=snr_gap_val,
        snr_key=snr_key,
        max_retries=max_snr_gap_retries,
        return_params=return_params,
        seed=seed_val if snr_gap_seed is None else snr_gap_seed,
    )

    return train_dataset, val_dataset, val_energy_matched_dataset, val_snr_gap_dataset
