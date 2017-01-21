GPUDirect simple example.

1) Get source code of NVIDIA-Linux-x86_64-X.Y driver as installed in your systems.

2) Extract it in the gpudma project directory and create symbolic link "nvidia" 
on NVIDIA-Linux-x86_64-X.Y driver directory.

3) Build NVIDIA driver in nvidia/kernel. We need only Module.symvers file from 
nvidia/kernel directory.

4) Build gpumem module.

5) Build application.


At this moment application create CUDA context and allocate GPU memory.
This memory pointer passed to gpumem module. Gpumem module get address of all physical 
pages of the allocates area and GPU page size. Application can get addresses and do mmap(), 
fill data pattern and free all of them. Than release GPU memory allocation and unlock pages.
