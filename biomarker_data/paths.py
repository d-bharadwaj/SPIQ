from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def sample_data_dir(n_qubits: int, *, use_copy: bool = True) -> Path:
    name = f"sampled_{n_qubits}_features_subproblem_1"
    if use_copy:
        name += " copy"
    return _PKG_DIR / name


def biomarker_pickle_dir() -> Path:
    return _PKG_DIR / "biomarker_pickle_data"
