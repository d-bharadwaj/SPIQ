from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def sample_data_dir(n_qubits: int) -> Path:
    """Directory with sampled features and PCBO coefficient arrays for ``n_qubits``."""
    return _PKG_DIR / "samples" / f"n{n_qubits}"


def biomarker_pickle_dir() -> Path:
    return _PKG_DIR / "biomarker_pickle_data"
