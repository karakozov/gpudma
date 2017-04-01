


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
 
#include "task_data.h"



/**
 * 		\brief	CUDA kernel for check buffer
 *
 * 		\param	sharedMemory 	area for exchange status information with host
 * 		\param	nbuf			number of buffer
 *
 *
 */
__global__ void checkCounterKernel( long *sharedMemory, int nbuf )
{

	__shared__ int shFlagIrq;

	TaskMonitor *ptrMonitor = (TaskMonitor*)sharedMemory;
	TaskBufferStatus *ts=(TaskBufferStatus *)sharedMemory;
	ts+=nbuf;

	uint64_t step = TaskCounts;
	int size=ts->sizeOfKBytes;
	int cnt=1024/8*size/step;

	uint64_t expect_data=nbuf*1024*size/8;
	expect_data += threadIdx.x;

	uint64_t *src = (uint64_t*)(ts->ptrCudaIn);
	src+=threadIdx.x;

	uint64_t *dst;

	TaskCheckData* check= &(ts->check[threadIdx.x]);

	unsigned int totalErrorForBuf=0;
	unsigned int errorCnt=0;
	unsigned int block_rd=0;
	unsigned int block_ok=0;
	unsigned int block_error=0;

	unsigned int flagError=0;

	TaskHostStatus *ptrHostStatus = ts->ptrHostStatus;
	shFlagIrq=0;


	//printf( "src=%p  x=%d y=%d z=%d expect_data=0x%.8lX\n", src, threadIdx.x, threadIdx.y, threadIdx.z, expect_data );


	for( int loop=0; ; loop++ )
	{
		if( 1==ptrMonitor->flagExit )
		{
			break;
		}

		if( 0==threadIdx.x )
			shFlagIrq=ts->irqFlag;


		if( 1!=shFlagIrq )
		{
			for( volatile int jj=0; jj<1000; jj++ );

			continue;
		}

		src = (uint64_t*)(ts->ptrCudaIn);
		src+=threadIdx.x;

		__syncthreads();


		flagError=0;
		check->flagError=1;

		if( 0==threadIdx.x )
		{

			dst=(uint64_t*)(ts->ptrCudaOut);
			dst+= ts->indexWr * cnt;

			for( int ii=0; ii<cnt; ii++ )
			{
				uint64_t	val;
				val = *src; src+=step;

				*dst++ = val;

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

			{
				int n=ts->indexWr+1;
				if( n==ts->indexMax )
					n=0;
				ts->indexWr=n;
				ptrHostStatus->indexWr=n;
			}

		} else
		{
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

		}


		check->flagError=flagError;
		check->cntError=errorCnt;

		if( 0==threadIdx.x )
		  ptrMonitor->block[nbuf].irqFlag=0;

		expect_data += 2*1024*size/8;

		__syncthreads();

		block_rd++;

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

			ts->blockRd=block_rd;
			ts->blockOk=block_ok;
			ts->blockError=block_error;
			//printf( "buf: %d  expect_data= 0x%.8lX \n", nbuf, expect_data );
		}

	}


}

/**
 * 		\brief	start checkCounterKernel
 *
 * 		\param	sharedMemory	pointer in CUDA memory of shared data
 * 		\param	nbuf			number of buffer
 * 		\param	stream			CUDA stream for this kernel
 *
 */
int run_checkCounter( long *sharedMemory, int nbuf, cudaStream_t& stream  )
{

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(1, 1);
    dim3 dimBlock(TaskCounts, 1, 1);
    checkCounterKernel<<<dimGrid, dimBlock, 0, stream>>>( sharedMemory, nbuf );

   return 0;
}


//__global__ void MonitorKernel( long* sharedMemory,  int nbuf, unsigned int index_rd  )
//{
//
//	TaskMonitor *ptrMonitor = (TaskMonitor*)sharedMemory;
//	TaskBufferStatus *ts=(TaskBufferStatus *)sharedMemory;
//	ts+=nbuf;
//
//	for( int loop=0; ; loop++ )
//	{
//		if( 1==ptrMonitor->flagExit )
//		{
//			break;
//		}
//
//		if( index_rd!=ptrMonitor->block[0].indexWr )
//			break;
//
//		for( volatile int jj=0; jj<10000; jj++ );
//	}
//
//
//}
//
//int run_Monitor(  long* sharedMemory, int nbuf, unsigned int index_rd, cudaStream_t stream )
//{
//
//    //Kernel configuration, where a two-dimensional grid and
//    //three-dimensional blocks are configured.
//    dim3 dimGrid(1, 1);
//    dim3 dimBlock(1, 1, 1);
//    MonitorKernel<<<dimGrid, dimBlock, 0, stream>>>(sharedMemory, nbuf, index_rd );
//
//
//}
