/*
 * TF_TestCnt.cpp
 *
 *  Created on: Jan 29, 2017
 *      Author: Dmitry Smekhov
 */

#include <sys/types.h>
#include <sys/stat.h>
#include "stdio.h"

#include <unistd.h>

#include "tf_testcnt.h"
#include "cl_cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "task_data.h"





TF_TestCnt::TF_TestCnt( int argc, char **argv ) : TF_TestThread( argc, argv )
{


	m_CountOfCycle   = GetFromCommnadLine( argc, argv, "-count", 16 );
	m_SizeBufferOfKb = GetFromCommnadLine( argc, argv, "-size", 256);



	td = new TaskData;


	m_pCuda=NULL;

	m_argc=argc;
	m_argv=argv;

}

TF_TestCnt::~TF_TestCnt() {

	delete m_pCuda; m_pCuda=NULL;
	delete td; 		td=NULL;
}


/**
 * 		\brief	Display current information about cheking buffers
 *
 * 		Function display information if 0==m_CountOfCycle
 * 		function is called from main with interval of 100 ms
 */
void	TF_TestCnt::StepTable( void )
{


	if( 0!=m_CountOfCycle )
		return;

	unsigned blockRd=0;
	unsigned blockOk=0;
	unsigned blockError=0;

	for( int ii=0; ii<3; ii++ )
	{
		blockRd+=td->ptrMonitor->block[ii].blockRd;
		blockOk+=td->ptrMonitor->block[ii].blockOk;
		blockError+=td->ptrMonitor->block[ii].blockError;

	}

	printf( "  %10d  %10d   %10d \r", blockRd, blockOk, blockError );
}

/**
 * 		\brief	Prepare CUDA and buffers
 *
 * 		Open CUDA device
 * 		Allocate three buffers and buffer for monitor
 */
void TF_TestCnt::PrepareInThread( void )
{

	m_pCuda = new CL_Cuda( m_argc, m_argv );

	td->monitor.id=100;


	td->countOfBuffers=3;
	int size=m_SizeBufferOfKb;
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

		td->ptrMonitor->block[ii].irqFlag=0;
		td->ptrMonitor->block[ii].blockOk=0;
		td->ptrMonitor->block[ii].blockError=0;
		td->ptrMonitor->block[ii].blockRd=0;
		for( int jj=0; jj<TaskCounts; jj++ )
		{
		 td->ptrMonitor->block[ii].check[jj].cntError=0;
		 td->ptrMonitor->block[ii].check[jj].flagError=0;
		}
	}

	td->ptrMonitor->flagExit=0;
	td->ptrMonitor->sig=0xAA24;


	printf( "m_CountOfCycle=%d\n", m_CountOfCycle );
	printf( "m_SizeBufferOfKb=%d [kB]\n\n", m_SizeBufferOfKb );

	if( 0==m_CountOfCycle )
		printf( "\n    BLOCK_RD    BLOCK_OK     BLOCK_ERROR \n" );

}

/**
 * 		\brief		Free buffers and close device
 *
 */
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

	size_t size=256*1024;


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


	int val;

	for( int kk=0; ; kk++ )
	{


		if( m_isTerminate || (m_CountOfCycle>0 && m_CountOfCycle==kk ))
		{
			td->ptrMonitor->flagExit=1;
			break;
		}

//		Check for checkCounter finished checking buffer 0
//		for( ; ; )
//		{
//		  val = td->ptrMonitor->block[0].irqFlag;
//		  if( 0==val )
//			  break;
//		}
		FillCounter( &td->bar1[0]);
		td->ptrMonitor->block[0].irqFlag=1;

		usleep( 100 );

//		Check for checkCounter finished checking buffer 1
//		for( ; ; )
//		{
//		  val = td->ptrMonitor->block[1].irqFlag;
//		  if( 0==val )
//			  break;
//		}
		FillCounter( &td->bar1[1]);
		td->ptrMonitor->block[1].irqFlag=1;

		usleep( 100 );

//		Check for checkCounter finished checking buffer 2
//		for( ; ; )
//		{
//		  val = td->ptrMonitor->block[2].irqFlag;
//		  if( 0==val )
//			  break;
//		}
		FillCounter( &td->bar1[2]);
		td->ptrMonitor->block[2].irqFlag=1;

		usleep( 100 );

	}

	for( volatile int jj=0; jj<100000000; jj++);

	td->ptrMonitor->flagExit=1;


	cudaStreamSynchronize( streamBuf0 );
	cudaStreamSynchronize( streamBuf1 );
	cudaStreamSynchronize( streamBuf2 );


	GetResult();

	return;



}

/**
 * 		\brief	Display result for all buffers
 *
 */
void TF_TestCnt::GetResult( void )
{
	GetResultBuffer( 0 );
	GetResultBuffer( 1 );
	GetResultBuffer( 2 );

}

/**
 * 		\brief	Display result for one buffers
 *
 * 		\param	nbuf	number of buffer
 *
 */
void TF_TestCnt::GetResultBuffer( int nbuf )
{

	TaskBufferStatus *ts=&(td->ptrMonitor->block[nbuf]);
	printf( "\nBuffer %d\n", nbuf );
	printf( "block_rd=%d\n", ts->blockRd );
	printf( "block_ok=%d\n", ts->blockOk );
	printf( "block_error=%d\n", ts->blockError );

	int flag_ok=1;
	for( int ii=0; ii<TaskCounts;ii++)
	{
		if( 0!=ts->check[ii].cntError )
		{
			flag_ok=0;
			break;
		}
	}

	if( 1==flag_ok )
	{
		printf( "Task 0:%d  - Ok\n",  TaskCounts-1 );

	} else
	{


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



}

