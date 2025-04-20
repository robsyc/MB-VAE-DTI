import os
from pathlib import Path
import sys

# Set the working directory to the root of the project
notebook_path = Path().resolve()
PROJECT_ROOT = notebook_path.parent

# Try to set the working directory, but handle the case where it's already set
try:
    assert PROJECT_ROOT.name == "MB-VAE-DTI", \
        f"Expected project root to be MB-VAE-DTI, got {PROJECT_ROOT.name}"
    print(f"Setting working directory to: {PROJECT_ROOT}")
    os.chdir(PROJECT_ROOT)
except AssertionError as e:
    # Check if we're already in the right directory
    current_dir = Path().resolve()
    if current_dir.name == "MB-VAE-DTI":
        print(f"Already in the correct directory: {current_dir}")
        PROJECT_ROOT = current_dir
    raise e  # Re-raise if we couldn't find the right directory

# Add project root to Python path to ensure imports work correctly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))