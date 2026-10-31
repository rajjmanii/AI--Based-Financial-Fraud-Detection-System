import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import make_synthetic_transactions  # noqa: E402
from src.models.model_registry import load_bundle  # noqa: E402
from src.models.train import run_training  # noqa: E402


@pytest.fixture
def synthetic_df():
    return make_synthetic_transactions(n_rows=3000, fraud_rate=0.03, seed=5)


@lru_cache(maxsize=1)
def _train_once():
    """Train once per test session on small synthetic data (fast, no Kaggle file needed)."""
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix="fraudshield_test_"))
    metadata = run_training(make_synthetic_transactions(3000, 0.03, seed=5), directory,
                            "synthetic test data", is_demo=True, log=lambda *_: None)
    return directory, metadata


@pytest.fixture
def trained_dir():
    return _train_once()[0]


@pytest.fixture
def metadata():
    return _train_once()[1]


@pytest.fixture
def bundle(trained_dir):
    return load_bundle(trained_dir)
