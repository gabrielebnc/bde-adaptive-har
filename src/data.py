"""Download full UCI archive, load inertial windows, and split whole subjects."""
import argparse
import hashlib
import math
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import seed_worker, write_json

DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x",
            "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]


def _extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains a symbolic link")
        zf.extractall(destination)


def dataset_directory(root, download: bool = False) -> Path:
    root = Path(root)
    directory = root / "UCI HAR Dataset"
    if (directory / "train/Inertial Signals/body_acc_x_train.txt").exists():
        return directory
    if not download:
        raise FileNotFoundError(f"Dataset missing at {directory}. Run python -m src.data --download --data-dir {root}")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "uci_har_complete.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading complete UCI archive to {archive}", flush=True)
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "wearable-har/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            with zipfile.ZipFile(partial) as zf:
                if zf.testzip() is not None:
                    raise ValueError("Downloaded archive failed CRC validation")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    _extract(archive, root)
    # The current UCI download wraps the original dataset ZIP in another ZIP.
    if not directory.exists():
        nested = list(root.rglob("UCI HAR Dataset.zip"))
        if len(nested) != 1:
            raise FileNotFoundError("Could not locate UCI HAR Dataset in the complete archive")
        _extract(nested[0], root)
    if not (directory / "test/Inertial Signals/body_acc_x_test.txt").exists():
        raise FileNotFoundError("Complete inertial signal files were not extracted")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    write_json(root / "download.json", {"url": DATASET_URL, "sha256": digest})
    return directory


def load_split(directory: Path, split: str):
    if split not in {"train", "test"}:
        raise ValueError("Expected official train or test split")
    signals = [np.loadtxt(directory / split / "Inertial Signals" / f"{name}_{split}.txt",
                          dtype=np.float32, ndmin=2) for name in CHANNELS]
    x = np.stack(signals, axis=1)
    y = np.loadtxt(directory / split / f"y_{split}.txt", dtype=np.int64, ndmin=1) - 1
    subjects = np.loadtxt(directory / split / f"subject_{split}.txt", dtype=np.int64, ndmin=1)
    if x.shape != (len(y), 9, 128) or len(subjects) != len(y):
        raise ValueError(f"Unexpected {split} shapes: {x.shape}, {y.shape}, {subjects.shape}")
    if not np.isfinite(x).all() or not np.isin(y, np.arange(6)).all():
        raise ValueError("Dataset contains invalid signals or labels")
    return x, y, subjects


def subject_masks(subjects, val_fraction=0.2, seed=42, val_subjects=None):
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("At least two subjects are needed")
    if val_subjects is None:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between zero and one")
        count = min(len(unique) - 1, max(1, math.ceil(val_fraction * len(unique))))
        val_subjects = np.random.default_rng(seed).permutation(unique)[:count].tolist()
    if not val_subjects or not set(val_subjects).issubset(set(unique)):
        raise ValueError("Validation subjects must be a nonempty subset of official training subjects")
    validation = np.isin(subjects, val_subjects)
    if validation.all():
        raise ValueError("Validation cannot contain every training subject")
    return ~validation, validation


def channel_statistics(x):
    mean = x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def normalize(x, mean, std):
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mean.shape != (9,) or std.shape != (9,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Normalization requires nine finite means and positive standard deviations")
    return (x - mean[None, :, None]) / std[None, :, None]


@dataclass
class DataBundle:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    metadata: dict


def prepare_data(root="data", *, download=False, val_fraction=0.2, seed=42,
                 val_subjects=None, normalization=None) -> DataBundle:
    directory = dataset_directory(root, download)
    x, y, subjects = load_split(directory, "train")
    test_x, test_y, test_subjects = load_split(directory, "test")
    if set(subjects) & set(test_subjects):
        raise ValueError("Official train/test subjects overlap")
    train_mask, val_mask = subject_masks(subjects, val_fraction, seed, val_subjects)
    if normalization is None:
        mean, std = channel_statistics(x[train_mask])
    else:
        mean, std = normalization["mean"], normalization["std"]
    normalized = normalize(x, mean, std)
    metadata = {
        "seed": seed, "channels": CHANNELS, "class_names": CLASS_NAMES,
        "source": "synthetic_test_fixture" if (directory / "SYNTHETIC_TEST_FIXTURE").exists() else "uci_har",
        "train_subjects": np.unique(subjects[train_mask]).tolist(),
        "validation_subjects": np.unique(subjects[val_mask]).tolist(),
        "test_subjects": np.unique(test_subjects).tolist(),
        "normalization": {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()},
        "class_counts": {name: np.bincount(labels, minlength=6).tolist() for name, labels in
                         [("train", y[train_mask]), ("validation", y[val_mask]), ("test", test_y)]},
        "sizes": {"train": int(train_mask.sum()), "validation": int(val_mask.sum()), "test": len(test_y)},
    }
    def dataset(features, labels):
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(features)), torch.from_numpy(labels.copy()))
    return DataBundle(dataset(normalized[train_mask], y[train_mask]),
                      dataset(normalized[val_mask], y[val_mask]),
                      dataset(normalize(test_x, mean, std), test_y), metadata)


def make_loader(dataset, batch_size=64, shuffle=False, seed=42, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
                      num_workers=num_workers, pin_memory=pin_memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-subjects", nargs="+", type=int)
    args = parser.parse_args()
    bundle = prepare_data(args.data_dir, download=args.download, seed=args.seed,
                          val_fraction=args.val_fraction, val_subjects=args.val_subjects)
    write_json(Path(args.data_dir) / "preparation.json", bundle.metadata)
    print(bundle.metadata)


if __name__ == "__main__":
    main()
