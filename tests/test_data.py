import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from mlops.data import prepare_splits, validate_frame
from mlops.pipeline import passes_gate


def test_split_is_reproducible_disjoint_and_stratified():
    frame = load_breast_cancer(as_frame=True).frame
    first, second = prepare_splits(frame), prepare_splits(frame)
    assert [len(first[k]) for k in ('train', 'validation', 'test')] == [341, 114, 114]
    sets = [set(part.index) for part in first.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(set.union(*sets)) == 569
    for name, part in first.items():
        assert part.equals(second[name])
        assert abs(part.target.mean() - frame.target.mean()) < 0.02


@pytest.mark.parametrize('corruption', ['nan', 'inf', 'target', 'column', 'duplicate'])
def test_invalid_dataset_is_rejected(corruption):
    frame = load_breast_cancer(as_frame=True).frame
    if corruption == 'nan':
        frame.iloc[0, 0] = np.nan
    elif corruption == 'inf':
        frame.iloc[0, 0] = np.inf
    elif corruption == 'target':
        frame.loc[0, 'target'] = 2
    elif corruption == 'column':
        frame = frame.drop(columns=frame.columns[0])
    else:
        frame.iloc[1] = frame.iloc[0]
    with pytest.raises(ValueError):
        validate_frame(frame)


@pytest.mark.parametrize('metrics,champion,expected', [
    ({'validation_roc_auc': 0.98, 'validation_accuracy': 0.96}, None, True),
    ({'validation_roc_auc': 0.90, 'validation_accuracy': 0.96}, None, False),
    ({'validation_roc_auc': 0.98, 'validation_accuracy': 0.80}, None, False),
    ({'validation_roc_auc': 0.98, 'validation_accuracy': 0.96}, 0.99, False),
    ({'validation_roc_auc': 0.98, 'validation_accuracy': 0.96}, 0.98, True),
    ({'validation_roc_auc': float('nan'), 'validation_accuracy': 0.96}, None, False),
])
def test_promotion_requires_quality_and_no_regression(metrics, champion, expected):
    assert passes_gate(metrics, champion, min_auc=0.95, min_accuracy=0.90) is expected
