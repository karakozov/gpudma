# GPUDirect RDMA example.

## Install
1 Clone repo and get source code of NVIDIA-Linux-x86_64-X.Y driver 
the same version as installed in your systems.

2 Extract it in the gpudma project directory and create symbolic link "nvidia" on NVIDIA-Linux-x86_64-X.Y driver directory.
Default location is ~/gpudma; For another location you must set variable GPUDMA_DIR, for example: GPUDMA_DIR=/xprj/gpudma

3 Build NVIDIA driver in nvidia/kernel. We need only Module.symvers file from nvidia/kernel directory.

4 Build gpumem module.

5 Build application app

6 Build application app_template

**Linux commands:**

git clone https://github.com/karakozov/gpudma.git

cp ~/Downloads/NVIDIA-Linux-x86_64-367.57.run ~/gpudma

./NVIDIA-Linux-x86_64-367.57.run -x

ln -svf NVIDIA-Linux-x86_64-367.57 nvidia

cd ~/gpudma/nvidia/kernel && make

cd ~/gpudma/module && make

cd ~/gpudma/app && make

## Load driver

cd ~/gpudma/module && ./drvload.sh

Check driver: ls -l /dev/gpumem

crw-rw-rw-. 1 root root 10, 55 Apr  2 21:57 /dev/gpumem

## Run app example

cd ~/gpudma/app && ./gpu_direct

Application create CUDA context and allocate GPU memory.
This memory pointer passed to gpumem module. Gpumem module get address of all physical 
pages of the allocates area and GPU page size. Application can get addresses and do mmap(), 
fill data pattern and free all of them. Than release GPU memory allocation and unlock pages.

Test must be finished with message: "Test successful"

## Build and run app_template

app_template must be built with Nsight Eclipse Edition from NVIDIA.

Command line for launch:  **app_template** **-count** ncount **-size** nsize
* ncount - block counts for read, 0 - for infinity cycle; Default is 16;
* nsize  - size of one buffers in kbytes. Maximum size is 65536. Default is 256;

Main mode is infinity cycle (ncount=0). There are two command for launch application:
* run_cycle_1M - launch with buffers of 1 megabytes
* run_cycle_64M - launch with buffers of 64 megabytes

Infinity cycle must be executed only from console. Nsight Eclipse Edition cannot correct display status line with "\r" symbol. If you can do it then send me about it, please.
For launch application from Nsight Eclipse Edition use non-zero value for count argument. This is enough for debugging.

There are main executing stages:

1. Create exemplar TF_TestCnt - launch thread for working with CUDA

2. Prepare
  * Open device
  * Allocate three buffers with size <nsize> and map in the BAR1 - class CL_Cuda
  * Allocate 64 kbytes buffer for struct TaskMonitor
  * Allocate page-locked HOST memory for td->hostBuffer 
  * Allocate page-locked HOST memory for struct TaskHostMonitor

3. Launch main cycle - TF_TestCnt::Run()
  * Launch thread for filling buffers - TF_TestCnt::FillThreadStart()
  * Launch kernel for checking data - run_checkCounter()
  * Check flag in the host memory and start DMA transfer - cudaMemcpyAsync()
  * Check data: TestCnt::CheckHostData()
  * Measuring velosity of data transfer

4. Periodcal launch function TF_TestCnt::StepTable() from function main() for display status information. It is working only for infinity cycle mode. Function display several parameters:
  * CUDA_RD - number of received buffers to CUDA
  * CUDA_OK - number of correct buffers to CUDA
  * CUDA_ERR - number of incorrect buffers to CUDA
  * HOST_RD - number of received buffers to HOST
  * HOST_OK - number of correct buffers to HOST
  * HOST_ERR - number of incorrect buffers to HOST
  * E2C_CUR - current velosity of data transfer from external device to CUDA
  * E2C_AVR - avarage velosity of data transfer from external device to CUDA
  * C2H_CUR - current velosity of data transfer from CUDA to HOST
  * C2H_AVR - avarage velosity of data transfer from CUDA to HOST

5. Function run_checkCounter() launch wrap of 32 thread for checking data. 

  Thread 0 is difference from another:
   * Read ts->irqFlag in the global memory and write it in the local wrap memory.
   * Write checking data to output buffers

  Thread 0 and another threads :
   * Check flag ptrMonitor->flagExit and exit if it is set.
   * Check received data  
   * Write first 16 errors to struct "check"

6. Display result after exiting from main cycle - TF_TestCnt::GetResult()

7. Free memory

Some notes:
* app_template/create_doc.sh - create documentation via doxygen
* There is class CL_Cuda_private for internal data for CL_Cuda
* There is file task_data.h with structs:
  * TaskData - internal task for TF_TestCnt
  * TaskMonitor - struct for shared memory in the CUDA 
  * TaskHostMonitor - struct for shared memory in the HOST
  * TaskBufferStatus - struct for work with one buffer
  * TaskCheckData - struct for error data
  * const int TaskCounts=32 - number of threads in the wrap




