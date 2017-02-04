/*
 * cl_cuda.cpp
 *
 *  Created on: Feb 4, 2017
 *      Author: user52
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
#include <cuda_runtime.h>

#include "gpumemioctl.h"

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

class CL_Cuda_prived
{
	public:

    int 			devID;
    cudaDeviceProp 	props;

    int				fd;

};

CL_Cuda::CL_Cuda( int argc, char** argv )
{


    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    //Get GPU information
    checkCudaErrors(cudaGetDevice(& pd->devID));
    checkCudaErrors(cudaGetDeviceProperties(&pd->props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
    		pd->devID, pd->props.name, pd->props.major, pd->props.minor);


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
}

