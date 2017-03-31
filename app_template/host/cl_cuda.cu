/*
 * cl_cuda.cpp
 *
 *  Created on: Feb 4, 2017
 *      Author: Dmitry Smekhov
 */

#include "cl_cuda.h"

// System includes
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/uio.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

//#include "cuda.h"
//#include "cuda_runtime_api.h"

#include "gpumemioctl.h"

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

void checkError(CUresult status);
bool wasError(CUresult status);

/**
 * 	\brief	Private data for CL_Cuda class
 */
class CL_Cuda_private
{
	public:

    int 			devID;	//!< Id for CUDA device
    cudaDeviceProp 	props;	//!< attributes for CUDA device

    int				fd;		//!< description of gpumem driver

    CUdevice 		device;		//!< Descriptor CUDA device
    char 			name[256];	//!< Name of CUDA device
    int 			major, minor;	//!< Capability numbers;
    size_t 			global_mem;		//!< Size of memory on CUDA device
    CUcontext  		context;		//!< Contex for all cuda functions

};

/**
 * 	\brief	Constructor
 *
 * 	\param	argc	argc from main function
 * 	\param	argv	argv from main function
 */
CL_Cuda::CL_Cuda( int argc, char** argv )
{


	pd = new CL_Cuda_private();

	cudaDeviceReset();

	checkError(cuInit(0));

//	int total = 0;
//	cudaGetDeviceCount( &total );
//	fprintf(stderr, "Total devices: %d\n", total);
//
	pd->devID=0;
 	cudaSetDevice(pd->devID);

    int total = 0;
    checkError(cuDeviceGetCount(&total));
    fprintf(stderr, "Total devices: %d\n", total);


    checkError(cuDeviceGet(&pd->device, 0));


    checkError(cuDeviceGetName( pd->name, 256, pd->device));
    fprintf(stderr, "Select device: %s\n", pd->name);

    // get compute capabilities and the devicename
    pd->major = 0; pd->minor = 0;
    checkError( cuDeviceComputeCapability(&pd->major, &pd->minor, pd->device));
    fprintf(stderr, "Compute capability: %d.%d\n", pd->major, pd->minor);

    pd->global_mem = 0;
    checkError( cuDeviceTotalMem(&pd->global_mem, pd->device));
    fprintf(stderr, "Global memory: %llu MB\n", (unsigned long long)(pd->global_mem >> 20));
    if(pd->global_mem > (unsigned long long)4*1024*1024*1024L)
        fprintf(stderr, "64-bit Memory Address support\n");



    checkError(cuCtxCreate(&pd->context, 0, pd->device));
	//checkError(cuCtxGetCurrent(&pd->context));

    pd->devID=0;
    //cudaSetDevice(pd->devID);


    pd->fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
    if(pd->fd < 0)
    {
        printf("Error open file %s\n", "/dev/"GPUMEM_DRIVER_NAME);
        throw( "Error /dev/gpumem");
    }

}


CL_Cuda::~CL_Cuda()
{
	// TODO Auto-generated destructor stub
	delete pd; pd=NULL;
}


//! Allocate buffer in CUDA memory and map it in BAR1 space
void CL_Cuda::AllocateBar1Buffer( int sizeOfKb, BAR1_BUF *pAdr )
{

	size_t size = sizeOfKb * 1024;
    gpudma_lock_t lock;
    gpudma_state_t *state = 0;
    unsigned int flag = 1;
	CUdeviceptr dptr = 0;
	int statesize = 0;
	int res = -1;

	int thLevel=0; // Level of local throw

    try
	{

	if( 0xA00!=pAdr->state)
	{
		fprintf(stderr, "BAR1_BUF is busy. state=0x%.3X != 0xA00\n",  pAdr->state );
		throw(0);
	}
	pAdr->state=0xA01;

    CUresult status = cuMemAlloc(&dptr, size);
    if(wasError(status)) {
        throw(thLevel);
    }
    thLevel++;

    fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr);

    status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    if(wasError(status)) {
        throw(thLevel); //goto do_free_memory;
    }


    // TODO: add kernel driver interaction...
    lock.addr = dptr;
    lock.size = size;
    res = ioctl(pd->fd, IOCTL_GPUMEM_LOCK, &lock);
    if(res < 0) {
        fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
        throw(thLevel); // goto do_free_attr;
    }


    pAdr->phy_addr= new uint64_t[lock.page_count];
    pAdr->app_addr= new void*[lock.page_count];
    thLevel++;

    statesize = (lock.page_count*sizeof(uint64_t) + sizeof(struct gpudma_state_t));
    state = (struct gpudma_state_t*)malloc(statesize);
    if(!state) {
        throw(thLevel); // goto do_free_attr;
    }
    memset(state, 0, statesize);
    state->handle = lock.handle;
    state->page_count = lock.page_count;
    res = ioctl(pd->fd, IOCTL_GPUMEM_STATE, state);
    if(res < 0) {
        fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
        throw(thLevel);// goto do_unlock;
    }

    fprintf(stderr, "Page count 0x%lx\n", state->page_count);
    fprintf(stderr, "Page size 0x%lx\n", state->page_size);

    pAdr->page_count=state->page_count;
    pAdr->page_size=state->page_size;
    pAdr->cuda_addr=(void*)dptr;
    pAdr->sizeOfBytes=size;


    for(unsigned ii=0; ii<state->page_count; ii++) {
    	if( state->page_count<16 )
         fprintf(stderr, "%02d: 0x%lx\n", ii, state->pages[ii]);
        void* va = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, pd->fd, (off_t)state->pages[ii]);
        if(va == MAP_FAILED ) {
             fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
             va = 0;
             throw(thLevel);
        } else {
            //fprintf(stderr, "%s(): Physical Address 0x%lx -> Virtual Address %p\n", __FUNCTION__, state->pages[i], va);
        	pAdr->app_addr[ii]=va;
        	pAdr->phy_addr[ii]=state->pages[ii];
        }
    }
    pAdr->state=0xA05; // Success
    fprintf(stderr, "CL_Cuda::AllocateBar1Buffer() - buffer id=%d is allocated, size=%d kB \n",  pAdr->id, sizeOfKb );
	} catch( int n )
	{
		switch( n )
		{

		case 2:
			delete pAdr->phy_addr; pAdr->phy_addr=NULL;
			delete pAdr->app_addr; pAdr->app_addr=NULL;
		case 1:
			cuMemFree(dptr);
		default:
			pAdr->state=0xA00;
			break;
		}
		throw(0);
	} catch( ... )
	{
		throw( 0 );
	}
}

//! Release buffer from BAR1 space and from CUDA memory
void CL_Cuda::FreeBar1Buffer( BAR1_BUF *pAdr )
{

	if( 0xA05!=pAdr->state)
	{
		fprintf(stderr, "BAR1_BUF is not allocate. state=0x%.3X != 0xA05\n",  pAdr->state );
		throw(0);
	}
	pAdr->state = 0xA10;

	// unmap virtual address
	void *va;
	for(unsigned ii=0; ii<pAdr->page_count; ii++)
	{
		va=pAdr->app_addr[ii];
		munmap(va, pAdr->page_size);
		pAdr->app_addr[ii]=NULL;
	}

	// free CUDA memory
	cuMemFree((CUdeviceptr)(pAdr->cuda_addr));

	// free array
	delete pAdr->app_addr; pAdr->app_addr=NULL;
	delete pAdr->phy_addr; pAdr->phy_addr=NULL;

	// Set empty state of pAdr
	pAdr->state = 0xA00;
	fprintf(stderr, "CL_Cuda::FreeBar1Buffer() - buffer id=%d is cleared \n",  pAdr->id);

}


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
        throw(0);
    }
}

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
