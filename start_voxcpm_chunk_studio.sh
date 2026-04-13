#!/bin/bash

# Get the directory where the script is located and navigate there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Initialize Conda for use in this script
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Conda is not found in PATH. Please ensure Conda is installed and accessible."
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate the conda environment
conda activate voxcpm
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment 'voxcpm'."
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the python script
python "$SCRIPT_DIR/voxcpm_chunk_studio.py"
EXIT_CODE=$?

# If the script exits with an error, keep the terminal open so the user can read it
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "VoxCPM Chunk Studio exited with code $EXIT_CODE."
    read -p "Press Enter to exit..."
fi

exit $EXIT_CODE
