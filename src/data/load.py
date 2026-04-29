import pandas as pd
from pathlib import Path

def load_data(file_name: str) -> pd.DataFrame:

    ROOT_DIR = Path(__file__).resolve().parents[2]/"data/raw"

    df = pd.read_csv(ROOT_DIR/file_name)

    return df