#!/bin/bash




#module unuse /usr/local/modulefiles/live/eb/all /usr/local/modulefiles/live/noeb
#module use /usr/local/modulefiles/staging/eb-znver3/all/
#module load cuda/11.1-gcc-9.1.0

module load python/anaconda3
source $condaDotFile
conda create --name stego_env --clone pytorch-1.8.1
module load cuda/10.1
conda install -y packaging
module load use.dev
module load gcc/5.4.0
python setup.py install --cuda_ext --cpp_ext



#module load cuDNN/7.6.4.38-gcccuda-2019b
#module load python/anaconda3
#source $condaDotFile
#conda create --name MeshGraphormerJADE2 --clone pytorch-1.8.1
#conda activate MeshGraphormerJADE2
#module load cuda/10.1

