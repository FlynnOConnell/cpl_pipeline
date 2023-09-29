#!/bin/bash --login
set -e
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate $ENV_PREFIX
exec "$@"
