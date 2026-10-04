from pathlib import Path
import pandas as pd

def load_data():
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    PROJECT_ROOT = Path.cwd().parent

    DATA_PATH = PROJECT_ROOT / "data" / "assignment_initial_conditions.csv"

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("position_m", ascending=False).reset_index(drop=True)
    return df