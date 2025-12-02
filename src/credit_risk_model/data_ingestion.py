import duckdb
from pathlib import Path
from config import DATA_RAW, DATA_SAMPLE, CSV_FILES, DB_PATH


def get_csv_path(filename):
    """Auto-select full or sample CSV."""
    full_path = DATA_RAW / filename
    sample_path = DATA_SAMPLE / filename
    
    if full_path.exists():
        return full_path
    elif sample_path.exists():
        return sample_path
    else:
        raise FileNotFoundError(f"Neither {full_path} nor {sample_path} exists")


def load_csvs_to_duckdb(con):
    """Load all CSV files into DuckDB tables."""
    for file in CSV_FILES:
        table_name = file.replace(".csv", "")
        csv_path = get_csv_path(file)
        
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS 
            SELECT * FROM read_csv_auto('{csv_path}')
        """)
        print(f"Loaded: {table_name}")


def get_connection():
    """Get or create DuckDB connection."""
    return duckdb.connect(str(DB_PATH))



