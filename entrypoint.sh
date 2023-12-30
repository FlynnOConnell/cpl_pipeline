#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
IFS=$'\n\t' # Set IFS to newline and tab.

# Activate the conda environment.
#eval "$(conda shell.bash hook)"  # not working

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda update conda
conda activate cpl_pipeline

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
#exec python run.py