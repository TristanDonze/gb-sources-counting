import logging
import sys

import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.logger import setup_logging

setup_logging()

logger = logging.getLogger("ValidationDataset")

HDF5_READ_BLOCK_SIZE = 100_000


def _show_progress():
    return sys.stderr.isatty()


class _PrecomputedSNRValidationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        max_K: int = 10,
        max_samples: int | None = 100_000,
        noise: bool = True,
        seed: int | None = None,
        snr_key: str = "snr",
        return_params: bool = False,
        indices: np.ndarray | None = None,
    ):
        if max_K < 2:
            raise ValueError("max_K must be >= 2 for these validation datasets")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be strictly positive or None")

        self.dataset_path = dataset_path
        self.max_K = max_K
        self.max_samples = max_samples
        self.noise = noise
        self.seed = seed
        self.snr_key = snr_key
        self.return_params = return_params
        self.indices = indices
        self.rng = np.random.default_rng(seed)

        with h5py.File(self.dataset_path, "r") as f:
            if "params" not in f or self.snr_key not in f["params"]:
                available = list(f["params"].keys()) if "params" in f else []
                raise ValueError(
                    f"snr_key={self.snr_key!r} was not found. "
                    f"Available params: {', '.join(available)}"
                )

            total_waveforms = f["waveforms"].shape[0]
            selection = self._normalize_indices(indices, total_waveforms)
            self.source_hdf5_indices = self._source_hdf5_indices(selection, total_waveforms)
            self.snr_values = np.asarray(f["params"][self.snr_key][selection])
            self.total_waveforms = len(self.snr_values)
            self.length = self.total_waveforms if max_samples is None else int(max_samples)

        if self.total_waveforms < self.max_K:
            raise ValueError(
                f"max_K={self.max_K} cannot be greater than available "
                f"waveforms={self.total_waveforms}"
            )

        logger.info(
            "%s: using %s candidate waveforms and building %s mixtures",
            self.__class__.__name__,
            self.total_waveforms,
            self.length,
        )
        hdf5_mixture_indices, targets, stats = self._build_hdf5_mixtures()
        self.targets = targets
        self.stats = stats

        self._load_selected_waveforms_and_remap(hdf5_mixture_indices)
        self.stats["unique_waveforms_loaded"] = int(len(self.loaded_hdf5_indices))
        self._log_stats()

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
    def _source_hdf5_indices(selection, total_waveforms):
        if isinstance(selection, slice):
            return np.arange(total_waveforms, dtype=np.int64)[selection]
        return np.asarray(selection, dtype=np.int64)

    def __len__(self):
        return self.length

    def _sample_k(self):
        return int(self.rng.integers(2, self.max_K + 1))

    def _empty_mixture_arrays(self):
        mixture_indices = np.full((self.length, self.max_K), -1, dtype=np.int64)
        targets = np.zeros(self.length, dtype=np.int64)
        return mixture_indices, targets

    def _build_hdf5_mixtures(self):
        raise NotImplementedError

    def _load_selected_waveforms_and_remap(self, hdf5_mixture_indices):
        used_hdf5_indices = hdf5_mixture_indices[hdf5_mixture_indices >= 0]
        self.loaded_hdf5_indices = np.unique(used_hdf5_indices)

        logger.info(
            "%s: loading %s unique waveforms into memory",
            self.__class__.__name__,
            len(self.loaded_hdf5_indices),
        )
        with h5py.File(self.dataset_path, "r") as f:
            waveforms_ds = f["waveforms"]
            self.waveforms = np.empty(
                (len(self.loaded_hdf5_indices),) + waveforms_ds.shape[1:],
                dtype=waveforms_ds.dtype,
            )
            first_index = int(self.loaded_hdf5_indices[0])
            last_index = int(self.loaded_hdf5_indices[-1])
            first_block = (first_index // HDF5_READ_BLOCK_SIZE) * HDF5_READ_BLOCK_SIZE

            for block_start in tqdm(
                range(first_block, last_index + 1, HDF5_READ_BLOCK_SIZE),
                desc=f"Loading waveforms ({self.__class__.__name__})",
                unit="chunk",
                dynamic_ncols=True,
                disable=not _show_progress(),
            ):
                block_stop = min(block_start + HDF5_READ_BLOCK_SIZE, waveforms_ds.shape[0])
                left = int(
                    np.searchsorted(
                        self.loaded_hdf5_indices,
                        block_start,
                        side="left",
                    )
                )
                right = int(
                    np.searchsorted(
                        self.loaded_hdf5_indices,
                        block_stop,
                        side="left",
                    )
                )
                if left == right:
                    continue

                block = waveforms_ds[block_start:block_stop]
                local_offsets = self.loaded_hdf5_indices[left:right] - block_start
                self.waveforms[left:right] = block[local_offsets]

            self.attr_names = []
            if self.return_params:
                self.attr_names = list(f["params"].keys())
                for key, value in f["params"].items():
                    setattr(self, key, value[self.loaded_hdf5_indices])

        local_mixture_indices = np.full_like(hdf5_mixture_indices, -1)
        valid_mask = hdf5_mixture_indices >= 0
        local_mixture_indices[valid_mask] = np.searchsorted(
            self.loaded_hdf5_indices,
            hdf5_mixture_indices[valid_mask],
        )
        self.mixture_indices = local_mixture_indices

    def _noise_rng(self, idx):
        seed = 0 if self.seed is None else self.seed
        seed_seq = np.random.SeedSequence([seed, int(idx), 12345])
        return np.random.default_rng(seed_seq)

    def _build_padded_params(self, local_indices, k):
        params = {}
        for attr in self.attr_names:
            values = np.asarray(getattr(self, attr)[local_indices])
            padded_shape = (self.max_K,) + values.shape[1:]
            padded = np.zeros(padded_shape, dtype=values.dtype)
            padded[:k] = values
            params[attr] = padded

        params["source_mask"] = np.arange(self.max_K) < k
        return params

    def _log_stats(self):
        readable = ", ".join(
            f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in self.stats.items()
        )
        logger.info("%s stats: %s", self.__class__.__name__, readable)

    def __getitem__(self, idx):
        target = int(self.targets[idx])
        local_indices = self.mixture_indices[idx, :target]
        summed_waveforms = self.waveforms[local_indices].sum(axis=0)

        if self.noise:
            summed_waveforms = summed_waveforms + self._noise_rng(idx).normal(
                0,
                1,
                size=summed_waveforms.shape,
            )

        if self.return_params:
            return summed_waveforms, target, self._build_padded_params(local_indices, target)

        return summed_waveforms, target


class SNRGapValidationDataset(_PrecomputedSNRValidationDataset):
    def __init__(
        self,
        dataset_path: str,
        target_snr_gap: float,
        max_K: int = 10,
        max_samples: int | None = 100_000,
        noise: bool = True,
        seed: int | None = None,
        snr_key: str = "snr",
        return_params: bool = False,
        indices: np.ndarray | None = None,
        max_retries: int = 100,
        max_gap_error: float | None = None,
    ):
        if target_snr_gap <= 0:
            raise ValueError("target_snr_gap must be strictly positive")
        if max_retries <= 0:
            raise ValueError("max_retries must be strictly positive")
        if max_gap_error is not None and max_gap_error < 0:
            raise ValueError("max_gap_error must be non-negative or None")

        self.target_snr_gap = target_snr_gap
        self.max_retries = max_retries
        self.max_gap_error = max_gap_error
        super().__init__(
            dataset_path=dataset_path,
            max_K=max_K,
            max_samples=max_samples,
            noise=noise,
            seed=seed,
            snr_key=snr_key,
            return_params=return_params,
            indices=indices,
        )

    def _build_hdf5_mixtures(self):
        snr_range = float(np.max(self.snr_values) - np.min(self.snr_values))
        if self.target_snr_gap > snr_range:
            raise ValueError(
                f"target_snr_gap={self.target_snr_gap} is greater than the "
                f"available SNR range={snr_range:.6g}"
            )

        self._snr_sorted_positions = np.argsort(self.snr_values)
        self._snr_sorted_values = self.snr_values[self._snr_sorted_positions]
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

        hdf5_mixture_indices, targets = self._empty_mixture_arrays()
        targets[:] = self.rng.integers(2, self.max_K + 1, size=self.length)

        low_sorted_positions = self.rng.integers(0, self._valid_low_count, size=self.length)
        low_positions = self._snr_sorted_positions[low_sorted_positions]
        low_snr = self._snr_sorted_values[low_sorted_positions]
        target_high_snr = low_snr + self.target_snr_gap
        high_positions = self._closest_snr_positions(target_high_snr, low_positions)
        high_snr = self.snr_values[high_positions]

        effective_gaps = np.abs(high_snr - low_snr).astype(np.float32)
        if self.max_gap_error is not None:
            gap_error = np.abs(effective_gaps - self.target_snr_gap)
            if np.any(gap_error > self.max_gap_error):
                raise ValueError(
                    "Vectorized SNR gap construction found samples outside "
                    "max_gap_error. Increase max_gap_error or disable it."
                )

        selected_positions = np.full((self.length, self.max_K), -1, dtype=np.int64)
        selected_positions[:, 0] = low_positions
        selected_positions[:, 1] = high_positions

        max_extra = self.max_K - 2
        if max_extra > 0:
            lower_snr = np.minimum(low_snr, high_snr)
            upper_snr = np.maximum(low_snr, high_snr)
            left = np.searchsorted(self._snr_sorted_values, lower_snr, side="left")
            right = np.searchsorted(self._snr_sorted_values, upper_snr, side="right")
            width = right - left
            if np.any(width < max_extra + 2):
                raise ValueError(
                    "Some SNR windows do not contain enough intermediate sources. "
                    "Try lowering max_K or increasing target_snr_gap."
                )

            random_offsets = (
                self.rng.random((self.length, max_extra)) * width[:, None]
            ).astype(np.int64)
            extra_sorted_positions = left[:, None] + random_offsets
            extra_positions = self._snr_sorted_positions[extra_sorted_positions]
            selected_positions[:, 2:] = extra_positions
            self._repair_duplicate_rows(selected_positions, targets, left, right)

        used_mask = np.arange(self.max_K)[None, :] < targets[:, None]
        hdf5_mixture_indices[used_mask] = self.source_hdf5_indices[
            selected_positions[used_mask]
        ]

        stats = {
            "target_snr_gap": float(self.target_snr_gap),
            "effective_gap_mean": float(np.mean(effective_gaps)),
            "effective_gap_std": float(np.std(effective_gaps)),
            "effective_gap_min": float(np.min(effective_gaps)),
            "effective_gap_max": float(np.max(effective_gaps)),
            "mean_retries": 0.0,
            "max_retries_used": 0,
        }
        return hdf5_mixture_indices, targets, stats

    def _closest_snr_positions(self, target_snr, forbidden_positions):
        insertion = np.searchsorted(self._snr_sorted_values, target_snr)
        left = np.clip(insertion - 1, 0, len(self._snr_sorted_values) - 1)
        right = np.clip(insertion, 0, len(self._snr_sorted_values) - 1)

        left_distance = np.abs(self._snr_sorted_values[left] - target_snr)
        right_distance = np.abs(self._snr_sorted_values[right] - target_snr)
        choose_right = right_distance <= left_distance
        sorted_positions = np.where(choose_right, right, left)
        positions = self._snr_sorted_positions[sorted_positions]

        same_as_forbidden = positions == forbidden_positions
        if np.any(same_as_forbidden):
            alt_sorted_positions = np.where(choose_right, left, right)
            positions[same_as_forbidden] = self._snr_sorted_positions[
                alt_sorted_positions[same_as_forbidden]
            ]

        return positions

    def _repair_duplicate_rows(self, selected_positions, targets, left, right):
        bad_rows = []
        for row in range(self.length):
            k = int(targets[row])
            values = selected_positions[row, :k]
            if len(np.unique(values)) != k:
                bad_rows.append(row)

        for row in bad_rows:
            k = int(targets[row])
            seen = set()
            for col in range(k):
                value = int(selected_positions[row, col])
                while value in seen:
                    sorted_position = int(self.rng.integers(left[row], right[row]))
                    value = int(self._snr_sorted_positions[sorted_position])
                selected_positions[row, col] = value
                seen.add(value)

class HomogeneousSNRValidationDataset(_PrecomputedSNRValidationDataset):
    def __init__(
        self,
        dataset_path: str,
        target_snr: float,
        pool_size: int = 500_000,
        max_K: int = 10,
        max_samples: int | None = 100_000,
        noise: bool = True,
        seed: int | None = None,
        snr_key: str = "snr",
        return_params: bool = False,
        indices: np.ndarray | None = None,
    ):
        if pool_size <= 0:
            raise ValueError("pool_size must be strictly positive")

        self.target_snr = target_snr
        self.pool_size = pool_size
        super().__init__(
            dataset_path=dataset_path,
            max_K=max_K,
            max_samples=max_samples,
            noise=noise,
            seed=seed,
            snr_key=snr_key,
            return_params=return_params,
            indices=indices,
        )

    def _build_hdf5_mixtures(self):
        pool_size = min(int(self.pool_size), self.total_waveforms)
        if pool_size < self.max_K:
            raise ValueError(
                f"pool_size={pool_size} must be at least max_K={self.max_K}"
            )

        distances = np.abs(self.snr_values - self.target_snr)
        if pool_size == self.total_waveforms:
            pool_positions = np.argsort(distances)
        else:
            pool_positions = np.argpartition(distances, pool_size - 1)[:pool_size]
            pool_positions = pool_positions[np.argsort(distances[pool_positions])]

        self._pool_positions = np.asarray(pool_positions, dtype=np.int64)
        pool_snr = self.snr_values[self._pool_positions]
        hdf5_mixture_indices, targets = self._empty_mixture_arrays()
        pool_state_by_k = {}

        for idx in tqdm(
            range(self.length),
            desc=f"Building mixtures ({self.__class__.__name__})",
            unit="mixture",
            dynamic_ncols=True,
            disable=not _show_progress(),
        ):
            target = self._sample_k()
            selected_positions = self._take_from_pool(target, pool_state_by_k)
            targets[idx] = target
            hdf5_mixture_indices[idx, :target] = self.source_hdf5_indices[selected_positions]

        stats = {
            "target_snr": float(self.target_snr),
            "pool_size": int(pool_size),
            "effective_snr_min": float(np.min(pool_snr)),
            "effective_snr_max": float(np.max(pool_snr)),
            "effective_snr_mean": float(np.mean(pool_snr)),
            "effective_snr_std": float(np.std(pool_snr)),
            "effective_abs_error_mean": float(np.mean(np.abs(pool_snr - self.target_snr))),
            "effective_abs_error_max": float(np.max(np.abs(pool_snr - self.target_snr))),
        }
        return hdf5_mixture_indices, targets, stats

    def _new_pool_permutation(self):
        return self.rng.permutation(self._pool_positions)

    def _take_from_pool(self, k, pool_state_by_k):
        state = pool_state_by_k.get(k)
        if state is None:
            state = {
                "permutation": self._new_pool_permutation(),
                "cursor": 0,
            }
            pool_state_by_k[k] = state

        if state["cursor"] + k > len(state["permutation"]):
            state["permutation"] = self._new_pool_permutation()
            state["cursor"] = 0

        start = state["cursor"]
        stop = start + k
        state["cursor"] = stop
        return state["permutation"][start:stop]
