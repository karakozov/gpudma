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


int init_cuda(int argc, char **argv);
int run_cuda( void );

TF_TestCnt::TF_TestCnt( int argc, char **argv )
{
	// TODO Auto-generated constructor stub

	m_isPrepareComplete=0;
	m_isComplete=0;
	m_isComplete=0;
	m_isTerminate=0;
	m_CycleCnt=0;

	m_pCuda=NULL;


	pthread_mutex_t		m_StartMutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t		m_StartCond  = PTHREAD_COND_INITIALIZER;

	m_argc=argc;
	m_argv=argv;
	//init_cuda( argc, argv );
}

TF_TestCnt::~TF_TestCnt() {

	delete m_pCuda; m_pCuda=NULL;
}



int 	TF_TestCnt::Prepare( int cnt )
{
		if( 0==cnt )
		{
		    int res = pthread_attr_init(&m_attrThread);
		    if(res != 0) {
		        fprintf(stderr, "%s\n", "Stream not started");
		        throw( "Stream not started" );
		    }

		    res = pthread_attr_setdetachstate(&m_attrThread, PTHREAD_CREATE_JOINABLE);
		    if(res != 0) {
		        fprintf(stderr, "%s\n", "Stream not started");
		        throw( "Stream not started" );
		    }

		    res = pthread_create(&m_hThread, &m_attrThread, ThreadFunc, this);
		    if(res != 0) {
		        fprintf(stderr, "%s\n", "Stream not started");
		        throw( "Stream not started" );
		    }
		}

		int ret=m_isPrepareComplete;
		if( ret )
		{
			printf( "Prepare - Ok\n");
			//fprintf( stderr, "Prepare - Ok\n");

		}

		return ret;
}

void* TF_TestCnt::ThreadFunc( void* lpvThreadParm )
{
	TF_TestCnt *test=(TF_TestCnt*)lpvThreadParm;
    void* ret;
    if( !test )
        return 0;
    ret=test->Execute();
    return ret;
}

void* TF_TestCnt::Execute( void )
{
		PrepareInThread();
		m_isPrepareComplete=1;

		// Wait for Start function
		pthread_mutex_lock( &m_StartMutex );
		pthread_cond_wait( &m_StartCond, &m_StartMutex );
		pthread_mutex_unlock( &m_StartMutex );

		fprintf( stderr, "Run\n");
		Run();

		CleanupInThread();

		m_isComplete=1;
		return NULL;
}

void	TF_TestCnt::Start( void )
{

	// Start Thread
	pthread_mutex_lock( &m_StartMutex );
	pthread_cond_signal( &m_StartCond );
	pthread_mutex_unlock( &m_StartMutex );
}

void 	TF_TestCnt::Stop( void )
{
	m_isTerminate=1;
	//fprintf( stderr, "%s - Ok\n", __FUNCTION__ );
}

int		TF_TestCnt::isComplete( void )
{
		return m_isComplete;
}

void	TF_TestCnt::StepTable( void )
{


	//m_CycleCnt++;
	//printf( "  %10d \r", m_CycleCnt );
	//Stop();
}

void TF_TestCnt::PrepareInThread( void )
{
	//init_cuda( m_argc, m_argv );

	m_pCuda = new CL_Cuda( m_argc, m_argv );

	m_Bar1[0].id=0;
	m_pCuda->AllocateBar1Buffer( 256, &m_Bar1[0] );

	fprintf( stderr, "%s - Ok\n", __FUNCTION__ );
}

void TF_TestCnt::CleanupInThread( void )
{

	m_pCuda->FreeBar1Buffer( &m_Bar1[0] );

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
	uint64_t val=m_CurrentCounter;

	for( int page=0; page<pBar1->page_count; page++ )
	{
		dst=(uint64_t*) (pBar1->app_addr[page]);
		for( int ii=size64; ii; ii--)
			*dst++=val++;

	}
	m_CurrentCounter=val;
}


int run_checkCounter( long* src, long* dst, int size );

void TF_TestCnt::Run( void )
{

	long *device_dst;
	size_t size=256*1024;
	cudaMalloc( (void**)(&device_dst), size );
	fprintf( stderr, "sizeof(long)=%d\n", sizeof(long));

	long *device_src=(long*)(m_Bar1[0].cuda_addr);

	long *host_dst =(long*)malloc(size);

	for( ; ; )
	{
		if( m_isTerminate )
			break;

		FillCounter( &m_Bar1[0]);
		//run_cuda();
		run_checkCounter( device_src, device_dst, size );

    	cudaMemcpy(host_dst, device_dst, size, cudaMemcpyDeviceToHost);
    	cudaDeviceSynchronize();

    	for( int ii; ii<16; ii++ )
    	{
    		fprintf( stderr, " 0x%.8X\n", host_dst[ii]);
    	}
		m_CycleCnt++;

		break;
	}

}
