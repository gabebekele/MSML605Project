import pytest
from src.thresholding import (
    compute_accuracy, compute_precision, 
    compute_recall, compute_f1_score
)

def test_metrics_calculation():
    # toy inputs
    conf = {"tp": 80, "tn": 70, "fp": 20, "fn": 30}
    
    # expected values 
    assert compute_accuracy(conf) == pytest.approx(0.75)
    assert compute_precision(conf) == pytest.approx(0.80)
    assert compute_recall(conf) == pytest.approx(0.7272, rel=1e-3)
    assert compute_f1_score(conf) == pytest.approx(0.7619, rel=1e-3)

def test_dividebyzero():
    # no images are same
    conf_empty = {"tp": 0, "tn": 100, "fp": 0, "fn": 100}
    
    assert compute_precision(conf_empty) == 0.0