from pathlib import Path

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW = BASE_DIR / "data"/"raw"
DATA_SAMPLE = BASE_DIR / "data"/"data_sample"
DATA_PROCESSED = BASE_DIR / "data"/"processed"

# Other directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
 
# Database file
DB_PATH = BASE_DIR / "home_credit.duckdb"

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_SAMPLE, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# CSV filenames
CSV_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv"
]



