import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = list(load_breast_cancer().feature_names)


def validate_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != FEATURES + ['target']:
        raise ValueError('Dataset must have the ordered 30 sklearn features and target')
    if len(frame) < 100 or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Dataset must have at least 100 finite rows')
    if set(frame.target.unique()) != {0, 1}:
        raise ValueError('Target must contain exactly the classes 0 and 1')
    if frame.duplicated(subset=FEATURES).any():
        raise ValueError('Duplicate observations would leak across splits')
    if frame.target.value_counts().min() < 20:
        raise ValueError('Insufficient samples in one target class')


def prepare_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_frame(frame)
    training, remaining = train_test_split(frame, test_size=0.4, random_state=SEED, stratify=frame.target)
    validation, test = train_test_split(remaining, test_size=0.5, random_state=SEED, stratify=remaining.target)
    return {'train': training, 'validation': validation, 'test': test}
