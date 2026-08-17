"""Shared path, data-loading, descriptor-schema, and output-safety helpers."""
import csv
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
MOLECULE_COLUMNS = ["Reactant1", "Reactant2", "Product", "Additive", "Solvent"]
RDKIT_DESCRIPTOR_SCHEMA = (
    DEFAULT_DATA_DIR / "extra-rdkit" / "train-rdkitfeature-Reactant1_feature_names.csv"
)

def load_rdkit_descriptor_names(schema_path: Path = RDKIT_DESCRIPTOR_SCHEMA) -> list[str]:
    """Load the exact 210-name descriptor schema used by the manuscript models."""
    schema_path = Path(schema_path)
    with schema_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != ["FeatureName"]:
        raise ValueError(f"Invalid RDKit descriptor schema header: {schema_path}")
    names = [row[0] for row in rows[1:] if row]
    if len(names) != 210 or len(set(names)) != 210:
        raise ValueError(
            f"Expected 210 unique RDKit descriptor names in {schema_path}, got {len(names)}"
        )
    return names

def calculate_manuscript_rdkit_descriptors(mol, names: list[str]) -> list[float]:
    """Calculate descriptors in the fixed manuscript order, independent of RDKit additions."""
    from rdkit.Chem import Descriptors
    functions = dict(Descriptors._descList)
    missing = [name for name in names if name not in functions]
    if missing:
        raise RuntimeError(
            "Installed RDKit is missing manuscript descriptors: " + ", ".join(missing)
        )
    return [float(functions[name](mol)) for name in names]

def load_training_data(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    round1 = pd.read_csv(data_dir / "round1_train_data.csv").copy()
    round2 = pd.read_csv(data_dir / "round2_train_data.csv").copy()
    if "rxntype" not in round1:
        round1["rxntype"] = 1
    if "rxntype" not in round2:
        raise ValueError("round2_train_data.csv must contain rxntype")
    return pd.concat([round1, round2], ignore_index=True)

def protected_outputs(paths, overwrite: bool) -> list[Path]:
    resolved = [Path(path).resolve() for path in paths]
    existing = [path for path in resolved if path.exists()]
    if existing and not overwrite:
        listing = "\\n  - ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing feature files. Use --overwrite only "
            "after making a backup:\\n  - " + listing
        )
    for path in resolved:
        path.parent.mkdir(parents=True, exist_ok=True)
    return resolved

def column_name(value: str) -> str:
    if value.isdigit():
        index = int(value)
        if not 0 <= index < len(MOLECULE_COLUMNS):
            raise ValueError("Column index must be between 0 and 4")
        return MOLECULE_COLUMNS[index]
    if value not in MOLECULE_COLUMNS:
        raise ValueError(f"Unknown molecular column: {value}")
    return value
