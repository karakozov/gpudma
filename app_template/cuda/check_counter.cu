/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
 
#include "task_data.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void checkCounterKernel( long* src, long* dst, int size )
{
//    printf("[%d, %d]:\t\tValue is:%d\n",\
//            blockIdx.y*gridDim.x+blockIdx.x,\
//            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//            sizeof(long));

	printf( "src=%p  dst=%p\n x=%d y=%d z=%d", src, dst, threadIdx.x, threadIdx.y, threadIdx.z );
	for( int ii=0; ii<16; ii++ )
	{
		//printf( "cuda: 0x%.8X \n", src[ii]);
	}
	long* ptr_dst=dst+threadIdx.x;
	long* ptr_src=src+threadIdx.x;
	int cnt=size/16;
	int step=2;
	long val = 0xAA0000 * threadIdx.x;
	for( int ii=0; ii<cnt; ii++ )
	{
		*ptr_dst=*ptr_src | val;
		ptr_dst+=step;
		ptr_src+=step;
	}
}


int run_checkCounter( long* src, long* dst, int size )
{

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(1, 1);
    dim3 dimBlock(2, 1, 1);
    checkCounterKernel<<<dimGrid, dimBlock>>>(src, dst, size );

    cudaDeviceSynchronize();
}


__global__ void MonitorKernel( long* src )
{
//    printf("[%d, %d]:\t\tValue is:%d\n",\
//            blockIdx.y*gridDim.x+blockIdx.x,\
//            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//            sizeof(long));

	printf( "Monitor start: src=%p  \n", src);

	TaskMonitor *ptrMonitor = (TaskMonitor*)src;
	for( int loop=0; ; loop++ )
	{
		if( 1==ptrMonitor->flagExit )
		{
			break;
		}

		if( 1==ptrMonitor->block0.irqFlag )
		{
			ptrMonitor->block0.irqFlag=2;
			ptrMonitor->block0.blockRd++;

		}
	}
	printf( "Monitor stop \n");


}

int run_Monitor( long* src, cudaStream_t stream )
{

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(1, 1);
    dim3 dimBlock(1, 1, 1);
    MonitorKernel<<<dimGrid, dimBlock, 0, stream>>>(src );


}
