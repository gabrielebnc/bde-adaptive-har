"""Small synthetic UCI-format files for offline correctness tests only."""
from pathlib import Path

import numpy as np

from src.data import CHANNELS


def write_uci_fixture(root, windows_per_subject=12):
    directory = Path(root) / "UCI HAR Dataset"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SYNTHETIC_TEST_FIXTURE").write_text("Synthetic correctness data, not UCI measurements.\n")
    rng = np.random.default_rng(7)
    for split, subject_ids in [("train", [1, 2, 3, 4, 5]), ("test", [6, 7])]:
        signals = directory / split / "Inertial Signals"
        signals.mkdir(parents=True)
        subjects = np.repeat(subject_ids, windows_per_subject)
        labels = np.tile(np.arange(windows_per_subject) % 6 + 1, len(subject_ids))
        np.savetxt(directory / split / f"y_{split}.txt", labels, fmt="%d")
        np.savetxt(directory / split / f"subject_{split}.txt", subjects, fmt="%d")
        for index, name in enumerate(CHANNELS):
            # Each channel and subject has a distinctive offset to detect leakage.
            values = rng.normal(size=(len(subjects), 128)) + index * 3 + subjects[:, None]
            np.savetxt(signals / f"{name}_{split}.txt", values, fmt="%.7f")
    return directory
