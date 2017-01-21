GPUDirect simple example.

1) Clone repo and get source code of NVIDIA-Linux-x86_64-X.Y driver 
the same version as installed in your systems.

2) Extract it in the gpudma project directory and create symbolic link "nvidia" 
on NVIDIA-Linux-x86_64-X.Y driver directory.

3) Build NVIDIA driver in nvidia/kernel. We need only Module.symvers file from 
nvidia/kernel directory.

4) Build gpumem module.

5) Build application.

Linux commands:

git clone https://github.com/karakozov/gpudma.git

cp ~/Downloads/NVIDIA-Linux-x86_64-367.57.run ~/gpudma

./NVIDIA-Linux-x86_64-367.57.run -x

ln -svf NVIDIA-Linux-x86_64-367.57 nvidia

cd ~/gpudma/nvidia/kernel && make

cd ~/gpudma/module && make

cd ~/gpudma/app && make


At this moment application create CUDA context and allocate GPU memory.
This memory pointer passed to gpumem module. Gpumem module get address of all physical 
pages of the allocates area and GPU page size. Application can get addresses and do mmap(), 
fill data pattern and free all of them. Than release GPU memory allocation and unlock pages.
