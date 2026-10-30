"""Phase 1: INPUT — Load the item dataset."""

from pathlib import Path
import pandas as pd

def load_dataset() -> pd.DataFrame:
    """Load the raw_skills.csv dataset.
    
    Returns a DataFrame with columns: 'Job Role', 'Skills'
    """
    csv_path = Path("data/raw_skills.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Please run 'python scripts/create_dataset.py' first."
        )
        
    return pd.read_csv(csv_path)