# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions


export PATH=/usr/local/packages/staging/eb-znver3/CUDAcore/11.1.1/bin:$PATH
#export PATH=/usr/local/packages/staging/eb-znver3/Ninja/1.10.2-GCCcore-11.2.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/packages/staging/eb-znver3/CUDAcore/11.1.1/lib64
export CUDA_HOME=/usr/local/packages/staging/eb-znver3/CUDAcore/11.1.1

