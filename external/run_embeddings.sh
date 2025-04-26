#!/bin/bash
# Generic script to activate a repository's virtual environment and run its embedding script
# Usage: ./run_embeddings.sh <repo_name> <h5_file>

# Exit on error
set -e

# Get absolute path to this script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <repo_name> <h5_file>"
    echo "Available repos: $(find $SCRIPT_DIR -maxdepth 1 -mindepth 1 -type d -not -name 'temp' -not -name '_*' -exec basename {} \; | tr '
' ' ')"
    exit 1
fi

REPO_NAME=$1
H5_FILE=$2

# Validate repo exists
REPO_DIR="${SCRIPT_DIR}/${REPO_NAME}"
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository '$REPO_NAME' not found in $SCRIPT_DIR"
    echo "Available repos: $(find $SCRIPT_DIR -maxdepth 1 -mindepth 1 -type d -not -name 'temp' -not -name '_*' -exec basename {} \; | tr '
' ' ')"
    exit 1
fi

# Define the path to the venv for this repo
VENV_PATH="${REPO_DIR}/venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if script.py exists
SCRIPT_PATH="${REPO_DIR}/script.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Validate h5 file exists
if [ ! -f "$H5_FILE" ]; then
    echo "Error: H5 file not found: $H5_FILE"
    exit 1
fi

# Build the command
CMD="python ${SCRIPT_PATH} --input \"$H5_FILE\""

# Activate the venv
echo "Activating virtual environment at $VENV_PATH"
source "${VENV_PATH}/bin/activate"

# Try to install requirements
echo "Checking for installation requirements..."
if [ -f "${REPO_DIR}/requirements.txt" ]; then
    echo "Installing from requirements.txt"
    pip install -qq -r "${REPO_DIR}/requirements.txt"
elif [ -f "${REPO_DIR}/setup.py" ]; then
    echo "Installing using setup.py"
    pip install -qq -e "${REPO_DIR}"
elif [ -f "${REPO_DIR}/pyproject.toml" ]; then
    echo "Installing using pyproject.toml"
    pip install -qq "${REPO_DIR}"
else
    echo "No requirements.txt, setup.py, or pyproject.toml found. Proceeding without additional installation."
fi

# Run the command
echo "Running: $CMD"
eval $CMD

# Check command status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Embedding generation completed successfully"
else
    echo "Embedding generation failed with status $STATUS"
fi

# Deactivate the venv
deactivate

exit $STATUS 