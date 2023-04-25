#!/bin/bash

module unuse /usr/local/modulefiles/live/eb/all /usr/local/modulefiles/live/noeb
module use /usr/local/modulefiles/staging/eb-znver3/all/
module load CUDAcore/11.1.1 GCC/11.2.0

module load Anaconda3

#module load cuDNN/7.6.4.38-gcccuda-2019b
