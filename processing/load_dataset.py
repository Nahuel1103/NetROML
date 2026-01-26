import pandas as pd
from pathlib import Path


def load_dataset(
    data_root: Path,
    building_id,
    filename_template="building_{building_id}_all_months.csv"
):
    """
    Load WiFi dataset using pathlib paths.
    """

    if not data_root.is_absolute():
        raise ValueError(f"data_root must be absolute: {data_root}")

    building_path = data_root / str(building_id) / ""
    csv_path = building_path / filename_template.format(
        building_id=building_id
    )

    print("csv_path: ", csv_path)
    print("building_path: ", building_path)

    if not building_path.exists():
        raise FileNotFoundError(f"Building folder not found: {building_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["mac_ap", "mac_cliente"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.dropna(subset=required_cols).reset_index(drop=True)
