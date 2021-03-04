# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

# module use /mnt/storage/easybuild/modules/local
module add languages/intel/2018-u3
module add languages/gcc/9.1.0
module add languages/anaconda2/5.3.1.tensorflow-1.12
module rm GCCcore/5.4.0 
module rm GCC/5.4.0-2.26
module rm binutils/2.26-GCCcore-5.4.0
module rm libs/cuda/9.0-gcc-5.4.0-2.26

export OMP_PLACES=cores
export OMP_PROC_BIND=close
