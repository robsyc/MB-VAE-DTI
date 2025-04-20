import pandas as pd
from pathlib import Path
import subprocess
from typing import Literal
import logging

# Configure logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding')

# Define paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

EXTERNAL_DIR = PROJECT_ROOT / "external"
TEMP_DIR = EXTERNAL_DIR / "temp"

# Ensure directories exists

TEMP_DIR.mkdir(exist_ok=True, parents=True)
assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
assert PROCESSED_DIR.exists(), f"Processed directory {PROCESSED_DIR} does not exist"
assert EXTERNAL_DIR.exists(), f"External directory {EXTERNAL_DIR} does not exist"

# Helper functions

def save_unique_representation_to_txt(
    df: pd.DataFrame,
    representation_column: Literal["Drug_SMILES", "Target_AA", "Target_DNA"],
    output_file_name: str
) -> Path:
    """
    Save unique drug or target string representation (to be used for embedding) to a text file.
    """
    entity_id = "Drug_ID" if representation_column == "Drug_SMILES" else "Target_ID"
    df = df.sort_values(by=entity_id)

    unique_representations = df[representation_column].unique()
    output_path = TEMP_DIR / output_file_name
    with open(output_path, "w") as f:
        for rep in unique_representations:
            f.write(f"{rep}\n")

    return output_path

def run_embedding_script(
    input_file_name: str,
    output_file_name: str,
    external_repo_name: Literal[
        "rdMorganFP", "biomed-multi-view",      # drugs
        "ESPF", "ESM", "nucleotide-transformer" # targets
    ]
) -> None:
    """
    Run the embedding script.
    """
    input_file_path = (TEMP_DIR / input_file_name).resolve()
    output_file_path = (TEMP_DIR / output_file_name).resolve()
    script_path = (EXTERNAL_DIR / "run_embeddings.sh").resolve()

    cmd = [
        str(script_path),
        external_repo_name,
        str(input_file_path),
        str(output_file_path)
    ]

    logger.info(f"Running embedding script with command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running embedding script: {e}")
        raise e
