set -e
conda_path=/shared/rejin/conda24
source $conda_path/etc/profile.d/conda.sh
conda activate base

eval ${@}
