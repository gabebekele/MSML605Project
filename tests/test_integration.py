import pytest
from src.evaluation import run_evaluation_pipeline

def test_pipeline_integration(tmp_path):
    """Simplest test to satisfy Milestone 2, Step 5.10."""
    # Use a temporary directory for outputs to keep the repo clean [cite: 44, 148]
    test_out = tmp_path / "test_outputs"
    
    config = {
        "data_dir": "data/lfw/downloads/extracted/TAR_GZ.ndownloader.figshare.com_files_5976018BV99nGMtc3Dm-0r8dGjUD5cMNKgNTG9Q_-xj9ajVNsA",
        "outputs_dir": str(test_out),
        "pairs_paths": {
            "val": "outputs/pairs/val.csv", 
            "test": "outputs/pairs/test.csv"
        }
    }

    # Run the actual pipeline logic on a tiny subset 
    try:
        run_evaluation_pipeline(config, "test_run", "none", "balanced_accuracy", limit=2)
        
        # Verify the two most important Milestone 2 artifacts exist [cite: 39, 191]
        assert (test_out / "runs" / "test_run" / "roc_curve.png").exists()
        assert (test_out / "runs" / "test_run" / "run.json").exists()
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")