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

	unsigned blockRd = td->ptrMonitor->block0.blockRd;
	printf( "  %10d \r", blockRd );

	//m_CycleCnt++;
	//printf( "  %10d \r", m_CycleCnt );
	//Stop();
}

void TF_TestCnt::PrepareInThread( void )
{
	//init_cuda( m_argc, m_argv );

	m_pCuda = new CL_Cuda( m_argc, m_argv );

	td->monitor.id=100;


	td->countOfBuffers=0;
	int size=256;
	td->sizeBufferOfBytes=size*1024;


	for( int ii=0; ii<td->countOfBuffers ; ii++ )
	{
		td->bar1[ii].id=ii;
		m_pCuda->AllocateBar1Buffer( size, &(td->bar1[ii]) );
	}
	m_pCuda->AllocateBar1Buffer( 256, &(td->monitor) );


	td->ptrMonitor=(TaskMonitor*)td->monitor.app_addr[0];

//	td->ptrMonitor->block0.ptrCudaIn=(void*)(td->bar1[0].cuda_addr);
//	td->ptrMonitor->block1.ptrCudaIn=(void*)(td->bar1[1].cuda_addr);
//	td->ptrMonitor->block2.ptrCudaIn=(void*)(td->bar1[2].cuda_addr);

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


int run_checkCounter( long* src, long* dst, int size );
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
//	size_t size=256*1024;
//	cudaMalloc( (void**)(&device_dst), size );
//	fprintf( stderr, "sizeof(long)=%d\n", sizeof(long));
//
//	long *device_src=(long*)(td->m_Bar1[0].cuda_addr);
//
//	long *host_dst =(long*)malloc(size);

	long *ptrCudaMonitor=(long*)(td->monitor.cuda_addr);

	cudaStream_t	streamMonitor;
	cudaStreamCreate( &streamMonitor );
	run_Monitor( ptrCudaMonitor, streamMonitor );



	for( ; ; )
	{
		if( m_isTerminate )
		{
			td->ptrMonitor->flagExit=1;
			break;
		}

		td->ptrMonitor->block0.irqFlag=1;

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

	cudaStreamSynchronize( streamMonitor );


}
