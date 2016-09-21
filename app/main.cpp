
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "gpudmaioctl.h"

#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

//-----------------------------------------------------------------------------

void checkError(CUresult status);
bool wasError(CUresult status);

//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    checkError(cuInit(0));

    int total = 0;
    checkError(cuDeviceGetCount(&total));
    fprintf(stderr, "Total devices: %d\n", total);

    CUdevice device;
    checkError(cuDeviceGet(&device, 0));

    char name[256];
    checkError(cuDeviceGetName(name, 256, device));
    fprintf(stderr, "Select device: %s\n", name);

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    checkError( cuDeviceComputeCapability(&major, &minor, device));
    fprintf(stderr, "Compute capability: %d.%d\n", major, minor);

    size_t global_mem = 0;
    checkError( cuDeviceTotalMem(&global_mem, device));
    fprintf(stderr, "Global memory: %llu MB\n", (unsigned long long)(global_mem >> 20));
    if(global_mem > (unsigned long long)4*1024*1024*1024L)
        fprintf(stderr, "64-bit Memory Address support\n");

    CUcontext  context;
    checkError(cuCtxCreate(&context, 0, device));

    size_t size = 0x100000;
    CUdeviceptr dptr = 0;
    unsigned int flag = 1;

    CUresult status = cuMemAlloc(&dptr, size);
    if(wasError(status)) {
        goto do_free_context;
    }

    fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr);

    status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    if(wasError(status)) {
        goto do_free_memory;
    }

    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens;
    status = cuPointerGetAttribute(&tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, dptr);
    if(wasError(status)) {
        goto do_free_attr;
    }

    // TODO: add kernel driver interaction...

do_free_attr:
    flag = 0;
    cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);

do_free_memory:
    cuMemFree(dptr);

do_free_context:
    cuCtxDestroy(context);

    return 0;
}

// -------------------------------------------------------------------

void checkError(CUresult status)
{
    if(status != CUDA_SUCCESS) {
        const char *perrstr = 0;
        CUresult ok = cuGetErrorString(status,&perrstr);
        if(ok == CUDA_SUCCESS) {
            if(perrstr) {
                fprintf(stderr, "info: %s\n", perrstr);
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        exit(0);
    }
}

//-----------------------------------------------------------------------------

bool wasError(CUresult status)
{
    if(status != CUDA_SUCCESS) {
        const char *perrstr = 0;
        CUresult ok = cuGetErrorString(status,&perrstr);
        if(ok == CUDA_SUCCESS) {
            if(perrstr) {
                fprintf(stderr, "info: %s\n", perrstr);
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        return true;
    }
    return false;
}

//-----------------------------------------------------------------------------