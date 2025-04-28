#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the absolute path of the directory containing this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the path to the main run_embeddings.sh script, relative to this script's location.
# Assumes 'scripts' and 'external' are sibling directories in the project root.
RUN_EMBEDDINGS_SCRIPT="${SCRIPT_DIR}/../external/run_embeddings.sh"

# Define the path to the temporary directory where HDF5 files are stored.
TEMP_DIR="${SCRIPT_DIR}/../external/temp"

# Define HDF5 file names (assuming they exist in TEMP_DIR)
SMILES_H5="dti_smiles.hdf5"
AA_H5="dti_aa.hdf5"
DNA_H5="dti_dna.hdf5"

echo "Starting embedding generation..."

# --- Add calls to run_embeddings.sh below ---

# echo "Running DTI embeddings..."

# echo "Running MorganFP embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "MorganFP" "${TEMP_DIR}/${SMILES_H5}"

# echo "Running ESPF embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "ESPF" "${TEMP_DIR}/${AA_H5}"

# echo "Running biomed-multi-view embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "biomed-multi-view" "${TEMP_DIR}/${SMILES_H5}"

# echo "Running ESM embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "ESM" "${TEMP_DIR}/${AA_H5}"

# echo "Running nucleotide-transformer embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "nucleotide-transformer" "${TEMP_DIR}/${DNA_H5}"

echo "Running pre-training embeddings..."

SMILES_H5="pretrain_smiles.hdf5"
AA_H5="pretrain_aa.hdf5"
DNA_H5="pretrain_dna.hdf5"

# echo "Running MorganFP embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "MorganFP" "${TEMP_DIR}/${SMILES_H5}"

echo "Running ESPF embeddings..."
bash "$RUN_EMBEDDINGS_SCRIPT" "ESPF" "${TEMP_DIR}/${AA_H5}"

# echo "Running biomed-multi-view embeddings..."
# bash "$RUN_EMBEDDINGS_SCRIPT" "biomed-multi-view" "${TEMP_DIR}/${SMILES_H5}"

echo "Running ESM embeddings..."
bash "$RUN_EMBEDDINGS_SCRIPT" "ESM" "${TEMP_DIR}/${AA_H5}"

echo "Running nucleotide-transformer embeddings..."
bash "$RUN_EMBEDDINGS_SCRIPT" "nucleotide-transformer" "${TEMP_DIR}/${DNA_H5}"

echo "All embedding scripts executed."
