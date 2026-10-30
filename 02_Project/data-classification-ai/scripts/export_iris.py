"""One-time exporter: writes the bundled Iris dataset to data/iris.csv.

Run once after setup:
    python scripts/export_iris.py
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def main() -> None:
    out_path = Path("data") / "iris.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=[f.replace(" (cm)", "") for f in iris.feature_names],
    )
    df["species"] = [iris.target_names[i] for i in iris.target]

    df.to_csv(out_path, index=False)

    print(f"[OK] Wrote {len(df)} rows x {len(df.columns)} columns to {out_path}")
    print(f"     Classes: {sorted(df['species'].unique().tolist())}")


if __name__ == "__main__":
    main()