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

__global__ void checkCounterKernel( long *ptrMonitor, int nbuf )
{
//    printf("[%d, %d]:\t\tValue is:%d\n",\
//            blockIdx.y*gridDim.x+blockIdx.x,\
//            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//            sizeof(long));

//	for( int ii=0; ii<16; ii++ )
//	{
//		//printf( "cuda: 0x%.8X \n", src[ii]);
//	}
//	long* ptr_dst=dst+threadIdx.x;
//	long* ptr_src=src+threadIdx.x;
//	int cnt=size/16;
//	int step=2;
//	long val = 0xAA0000 * threadIdx.x;
//	for( int ii=0; ii<cnt; ii++ )
//	{
//		*ptr_dst=*ptr_src | val;
//		ptr_dst+=step;
//		ptr_src+=step;
//	}

	TaskBufferStatus *ts=(TaskBufferStatus *)ptrMonitor;
	ts+=nbuf;

	uint64_t step = TaskCounts;
	int size=ts->sizeOfKBytes;
	int cnt=1024/8*size/step;

	uint64_t expect_data=nbuf*1024*size/8;
	expect_data += threadIdx.x;

	uint64_t *src = (uint64_t*)(ts->ptrCudaIn);
	src+=threadIdx.x;

	TaskCheckData* check= &(ts->check[threadIdx.x]);

	unsigned int totalErrorForBuf=0;
	unsigned int errorCnt=0;
	unsigned int block_rd=0;
	unsigned int block_ok=0;
	unsigned int block_error=0;

	unsigned int flagError=0;



	printf( "src=%p  x=%d y=%d z=%d expect_data=%ld\n", src, threadIdx.x, threadIdx.y, threadIdx.z, expect_data );

	flagError=0;
	check->flagError=1;
	for( int ii=0; ii<cnt; ii++ )
	{
		uint64_t	val;
		val = *src; src+=step;

		if( val!=expect_data )
		{
			if( errorCnt<16 )
			{
				check->nblock[errorCnt]=block_rd;
				check->adr[errorCnt]=ii;
				check->expect_data[errorCnt]=expect_data;
				check->receive_data[errorCnt]=val;
			}
			errorCnt++;
			flagError++;
		}
		expect_data+=step;
	}
	check->flagError=flagError;
	check->cntError=errorCnt;
	__syncthreads();

	if( 0==threadIdx.x )
	{
		// Check all task
		unsigned int flagErr=0;
		for( int ii=0; ii<TaskCounts; ii++ )
		{
			if( ts->check[ii].flagError )
			{
				flagErr=1;
			}
		}
		if( 0==flagErr)
		{
			block_ok++;
		} else
		{
			block_error++;
		}
		block_rd++;

		ts->blockRd=block_rd;
		ts->blockOk=block_ok;
		ts->blockError=block_error;

	}


}


int run_checkCounter( long *ptrMonitor, int nbuf, cudaStream_t& stream  )
{

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(1, 1);
    dim3 dimBlock(TaskCounts, 1, 1);
    checkCounterKernel<<<dimGrid, dimBlock, 0, stream>>>( ptrMonitor, nbuf );

   // cudaDeviceSynchronize();
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

		if( 1==ptrMonitor->block[0].irqFlag )
		{
			ptrMonitor->block[0].irqFlag=2;
			ptrMonitor->block[0].blockRd++;

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
