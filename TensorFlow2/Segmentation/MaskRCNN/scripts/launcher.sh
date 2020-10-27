#set -ex
set -e
conda_path=/shared/sboshin/conda
source $conda_path/etc/profile.d/conda.sh
conda activate base

eval ${@}
