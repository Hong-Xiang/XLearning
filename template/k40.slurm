#!/bin/bash

#SBATCH -J tensorflow
#SBATCH -p centos7k40
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=hxnikolatesla@sjtu.edu.cn
#SBATCH -o %j.out
#SBATCH -e %j.err

source /usr/share/Modules/init/bash
unset MODULEPATH
module use /lustre/usr/modulefiles/pi
source ~/utils/set_env.sh

echo "The job begins at `date`."

./work.sh

echo "The job completes at `date`."
