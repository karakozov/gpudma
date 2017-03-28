/*
 * TF_TestCnt.cpp
 *
 *  Created on: Jan 29, 2017
 *      Author: root
 */

#include <sys/types.h>
#include <sys/stat.h>
#include "stdio.h"

#include "tf_testcnt.h"
#include "cl_cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "task_data.h"

int init_cuda(int argc, char **argv);
int run_cuda( void );




TF_TestCnt::TF_TestCnt( int argc, char **argv ) : TF_TestThread( argc, argv )
{
	// TODO Auto-generated constructor stub

	td = new TaskData;


	m_pCuda=NULL;

	m_argc=argc;
	m_argv=argv;

}

TF_TestCnt::~TF_TestCnt() {

	delete m_pCuda; m_pCuda=NULL;
	delete td; 		td=NULL;
}



void	TF_TestCnt::StepTable( void )
{

	unsigned blockRd = td->ptrMonitor->block[0].blockRd;
	//printf( "  %10d \r", blockRd );

	//m_CycleCnt++;
	//printf( "  %10d \r", m_CycleCnt );
	//Stop();
}

void TF_TestCnt::PrepareInThread( void )
{
	//init_cuda( m_argc, m_argv );

	m_pCuda = new CL_Cuda( m_argc, m_argv );

	td->monitor.id=100;


	td->countOfBuffers=3;
	int size=256;
	td->sizeBufferOfBytes=size*1024;


	for( int ii=0; ii<td->countOfBuffers ; ii++ )
	{
		td->bar1[ii].id=ii;
		m_pCuda->AllocateBar1Buffer( size, &(td->bar1[ii]) );
	}
	m_pCuda->AllocateBar1Buffer( 256, &(td->monitor) );


	td->ptrMonitor=(TaskMonitor*)td->monitor.app_addr[0];

	for( int ii=0; ii<td->countOfBuffers; ii++ )
	{
		td->ptrMonitor->block[ii].ptrCudaIn=(void*)(td->bar1[ii].cuda_addr);

		td->ptrMonitor->block[ii].sizeOfKBytes=size;
	}

	td->ptrMonitor->flagExit=0;
	td->ptrMonitor->sig=0xAA24;


	fprintf( stderr, "%s - Ok\n", __FUNCTION__ );
}

void TF_TestCnt::CleanupInThread( void )
{

	for( int ii=0; ii<td->countOfBuffers; ii++ )
	{
		m_pCuda->FreeBar1Buffer( &(td->bar1[ii]) );
	}
	m_pCuda->FreeBar1Buffer( &(td->monitor) );

	delete m_pCuda; m_pCuda=NULL;

	fprintf( stderr, "%s - Ok\n", __FUNCTION__ );
}

/**
 * 	\brief	fill buffer
 *
 * 	\param	pBar1	description of buffer
 *
 * 	function fill bar1 buffer via pBar1->app_addr[]
 *
 */
void TF_TestCnt::FillCounter( CL_Cuda::BAR1_BUF *pBar1 )
{
	if( 0xA05 != pBar1->state )
		throw(0);

	int size64=pBar1->page_size/8;
	uint64_t *dst;
	uint64_t val=td->currentCounter;

	for( int page=0; page<pBar1->page_count; page++ )
	{
		dst=(uint64_t*) (pBar1->app_addr[page]);
		for( int ii=size64; ii; ii--)
			*dst++=val++;

	}
	td->currentCounter=val;
}


int run_checkCounter( long *ptrMonitor, int nbuf, cudaStream_t& stream );
int run_Monitor( long* src, cudaStream_t stream );

/**
 * 	\brief	Main working cycle
 *
 * 	It is main working cycle.
 * 	Function FillCounter  simulate to work external DMA channel.
 *
 */
void TF_TestCnt::Run( void )
{

//	long *device_dst;
	size_t size=256*1024;
//	cudaMalloc( (void**)(&device_dst), size );
//	fprintf( stderr, "sizeof(long)=%d\n", sizeof(long));
//
	long *device_src=(long*)(td->bar1[0].cuda_addr);
	long *device_dst=(long*)(td->bar1[1].cuda_addr);
//
//	long *host_dst =(long*)malloc(size);

	FillCounter( &td->bar1[0]);
	FillCounter( &td->bar1[1]);
	FillCounter( &td->bar1[2]);

	long *ptrCudaMonitor=(long*)(td->monitor.cuda_addr);

	cudaStream_t	streamBuf0;
	cudaStream_t	streamBuf1;
	cudaStream_t	streamBuf2;

	cudaStreamCreate( &streamBuf0 );
	cudaStreamCreate( &streamBuf1 );
	cudaStreamCreate( &streamBuf2 );


	run_checkCounter(  ptrCudaMonitor, 0, streamBuf0 );
	run_checkCounter(  ptrCudaMonitor, 1, streamBuf1 );
	run_checkCounter(  ptrCudaMonitor, 2, streamBuf2 );

	cudaStreamSynchronize( streamBuf0 );
	cudaStreamSynchronize( streamBuf1 );
	cudaStreamSynchronize( streamBuf2 );



	GetResult();

	return;


	cudaStream_t	streamMonitor;
	cudaStreamCreate( &streamMonitor );
//	run_Monitor( ptrCudaMonitor, streamMonitor );




	for( ; ; )
	{
		if( m_isTerminate )
		{
			td->ptrMonitor->flagExit=1;
			break;
		}

		td->ptrMonitor->block[0].irqFlag=1;

		for( volatile int jj=0; jj<100000000; jj++);

//		FillCounter( &td->m_Bar1[0]);
//		run_checkCounter( device_src, device_dst, size );
//
//    	cudaMemcpy(host_dst, device_dst, size, cudaMemcpyDeviceToHost);
//    	cudaDeviceSynchronize();
//
//    	for( int ii; ii<16; ii++ )
//    	{
//    		fprintf( stderr, " 0x%.8X\n", host_dst[ii]);
//    	}
//		m_CycleCnt++;


		//td->ptrMonitor->flagExit=1;
		//break;
	}

//	cudaStreamSynchronize( streamMonitor );


}

void TF_TestCnt::GetResult( void )
{
	GetResultBuffer( 0, &(td->ptrMonitor->block[0]) );
	GetResultBuffer( 1, &(td->ptrMonitor->block[1]) );
	GetResultBuffer( 2, &(td->ptrMonitor->block[2]) );

}

void TF_TestCnt::GetResultBuffer( int nbuf, TaskBufferStatus *ts )
{

	printf( "\nBuffer %d\n", nbuf );
	printf( "block_rd=%d\n", ts->blockRd );
	printf( "block_ok=%d\n", ts->blockOk );
	printf( "block_error=%d\n", ts->blockError );

	for( int ii=0; ii<TaskCounts; ii++ )
	{
		unsigned int cntError=ts->check[ii].cntError;
		if( 0==cntError )
		{
			printf( "Task %d -Ok\n", ii );
		} else
		{
			printf( "\nTask %d \n", ii );
			printf( "   cntError=%d\n", cntError);
			if( cntError>16 )
				cntError=16;
			for( int jj=0; jj<cntError; jj++ )
			{
			 printf( "%2d block: %4d  addr: 0x%.4X  receive: 0x%.8lX  expect: 0x%.8lX\n",
					 jj,
					 ts->check[ii].nblock[jj],
					 ts->check[ii].adr[jj],
					 ts->check[ii].receive_data[jj],
					 ts->check[ii].expect_data[jj]
			 	 );
			}
		}

	}



}

