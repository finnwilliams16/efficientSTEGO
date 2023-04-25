#!/bin/bash

#module load python/anaconda3
module load Anaconda3/5.3.0


module unuse /usr/local/modulefiles/live/eb/all /usr/local/modulefiles/live/noeb
module use /usr/local/modulefiles/staging/eb-znver3/all/
module load Anaconda3/2021.11 CUDAcore/11.1.1 GCC/9.5.0 
# module load Anaconda3/2021.11 CUDAcore/11.1.1 GCC/11.2.0
#module load Ninja/1.10.2-GCCcore-11.2.0
#which CUDAcore/11.1.1

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

