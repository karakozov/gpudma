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


	td->countOfCycle   = GetFromCommnadLine( argc, argv, "-count", 16 );
	td->sizeBufferOfKb = GetFromCommnadLine( argc, argv, "-size", 256);



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
 * 		Function display information if 0==td->countOfCycle
 * 		function is called from main with interval of 100 ms
 */
void	TF_TestCnt::StepTable( void )
{


	if( 0!=td->countOfCycle )
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
	int size=td->sizeBufferOfKb;
	td->sizeBufferOfBytes=size*1024;


	for( int ii=0; ii<td->countOfBuffers ; ii++ )
	{
		td->bar1[ii].id=ii;
		m_pCuda->AllocateBar1Buffer( size, &(td->bar1[ii]) );
	}
	m_pCuda->AllocateBar1Buffer( 256, &(td->monitor) );


	td->ptrMonitor=(TaskMonitor*)td->monitor.app_addr[0];

	unsigned int indexMax=2;
	size_t outSizeBlock=size*1024/TaskCounts; // size of output buffer
	int n=512*1024*1024 / outSizeBlock; // count blocks in 512 MB buffer

	size_t outSizeBuffer=n*outSizeBlock;

	td->outputSizeBuffer = outSizeBuffer;
	td->outputSizeBlock  = outSizeBlock;
	td->outputCountBlock = n;

	cudaError_t ret;

	void *ptr=(void*)td->hostBuffer;
	ret=cudaMallocHost( &ptr, outSizeBlock );
	if( cudaSuccess!=ret )
		throw( "Error page-locked memory allocate for hostBuffer" );

	ptr=(void*)td->hostMonitor;
	ret=cudaMallocHost( &ptr, 4096 );
	if( cudaSuccess!=ret )
		throw( "Error page-locked memory allocate for hostMonitor" );


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

		 td->ptrMonitor->block[ii].indexRd=0;
		 td->ptrMonitor->block[ii].indexWr=0;

		 td->ptrMonitor->block[ii].indexMax = indexMax;
		 ret= cudaMalloc( &(td->ptrMonitor->block[ii].ptrCudaOut), outSizeBuffer );
		 if( cudaSuccess != ret )
			 throw( "Error memory allocation for output buffer" );

	}

	td->ptrMonitor->flagExit=0;
	td->ptrMonitor->sig=0xAA24;


	printf( "td->countOfCycle=%d\n", td->countOfCycle );
	printf( "td->sizeBufferOfKb=%d [kB]\n\n", td->sizeBufferOfKb );

	if( 0==td->countOfCycle )
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

	cudaFreeHost( td->hostBuffer );
	cudaFreeHost( td->hostMonitor );

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


int run_checkCounter( long *sharedMemory, int nbuf, cudaStream_t& stream );
int run_Monitor( long* sharedMemory, int nbuf, unsigned int index_rd, cudaStream_t stream );

/**
 * 	\brief	Main working cycle
 *
 * 	It is main working cycle.
 * 	Function FillCounter  simulate to work external DMA channel.
 *
 */
void TF_TestCnt::Run( void )
{


	FillThreadStart();

	long *ptrCudaMonitor=(long*)(td->monitor.cuda_addr);

	cudaStream_t	streamBuf0;
	cudaStream_t	streamBuf1;
	cudaStream_t	streamBuf2;
	cudaStream_t	streamMonitor;
	cudaStream_t	streamDMA;

	cudaStreamCreate( &streamBuf0 );
	cudaStreamCreate( &streamBuf1 );
	cudaStreamCreate( &streamBuf2 );
	cudaStreamCreate( &streamMonitor );
	cudaStreamCreate( &streamDMA );


	run_checkCounter(  ptrCudaMonitor, 0, streamBuf0 );
	run_checkCounter(  ptrCudaMonitor, 1, streamBuf1 );
	run_checkCounter(  ptrCudaMonitor, 2, streamBuf2 );


	int val;
	int blockRd;

	int nbuf;
	unsigned int indexRd[3]={ 0, 0, 0 };

	cudaError_t ret;

	int status=0;


	for( int kk=0; ; kk++ )
	{

		blockRd=td->ptrMonitor->block[0].blockRd;

		if( m_isTerminate || (td->countOfCycle>0 && td->countOfCycle==blockRd ))
		{
			td->ptrMonitor->flagExit=1;
			break;
		}

		switch( status )
		{
			case 0: // run monitor
				run_Monitor(  ptrCudaMonitor, nbuf, indexRd[nbuf], streamMonitor );
				status=1;
				break;

			case 1: // wait for ready current buffer and start DMA  read
				ret=cudaStreamQuery( streamMonitor );
				if( cudaSuccess==ret )
				{

					uint64_t* d_src=(uint64_t*)(td->ptrMonitor->block[nbuf].ptrCudaOut);
					d_src+=indexRd[nbuf]*td->outputSizeBlock/8;

					cudaMemcpyAsync( td->hostBuffer, d_src, td->outputSizeBlock, cudaMemcpyDeviceToHost, streamDMA );
					status=2;
				}
			case 2: // wait for data transfer complete
				ret=cudaStreamQuery( streamDMA );
				if( cudaSuccess==ret )
				{
					CheckHostData( td->hostBuffer );

					int n=indexRd[nbuf]+1;
					if( n==td->outputCountBlock )
						n=0;
					indexRd[nbuf]=n;

					n=nbuf+1;
					if( 3==n )
						n=0;
					nbuf=n;

					status=0;
				}
		}



		//usleep( 100 );

	}

	usleep( 10000 );

	td->ptrMonitor->flagExit=1;


	cudaStreamSynchronize( streamBuf0 );
	cudaStreamSynchronize( streamBuf1 );
	cudaStreamSynchronize( streamBuf2 );


	GetResult();

	FillThreadDestroy();

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

void TF_TestCnt::FillThreadStart( void )
{
    int res = pthread_attr_init(&m_attrFillThread);
    if(res != 0) {
        fprintf(stderr, "%s\n", "Stream not started");
        throw( "Stream not started" );
    }

    res = pthread_attr_setdetachstate(&m_attrFillThread, PTHREAD_CREATE_JOINABLE);
    if(res != 0) {
        fprintf(stderr, "%s\n", "Stream not started");
        throw( "Stream not started" );
    }

    res = pthread_create(&m_hFillThread, &m_attrFillThread, FillThreadFunc, this);
    if(res != 0) {
        fprintf(stderr, "%s\n", "Stream not started");
        throw( "Stream not started" );
    }
}

void TF_TestCnt::FillThreadDestroy( void )
{

}


void* TF_TestCnt::FillThreadFunc( void* lpvThreadParm )
{
	TF_TestCnt *test=(TF_TestCnt*)lpvThreadParm;
    void* ret;
    if( !test )
        return 0;
    ret=test->FillExecute();
    return ret;
}

void* TF_TestCnt::FillExecute( void )
{

	//printf( "\nFillCounter Start\n");
	for( ; ; )
	{

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
		if( td->ptrMonitor->flagExit )
			break;


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
		if( td->ptrMonitor->flagExit )
			break;

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
		if( td->ptrMonitor->flagExit )
			break;

	}
	//printf( "\nFillCounter Stop\n");

	return NULL;
}


//! Check received data
void TF_TestCnt::CheckHostData( uint64_t* src )
{
	printf( "CheckHostData: 0x%.8lX \n", *src );
}
