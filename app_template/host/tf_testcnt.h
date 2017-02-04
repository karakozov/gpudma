/*
 * TF_TestCnt.h
 *
 *  Created on: Jan 29, 2017
 *      Author: root
 */

#ifndef TF_TESTCNT_H_
#define TF_TESTCNT_H_

#include <pthread.h>
#include "tf_test.h"

//class CL_Cuda;
//struct CL_Cuda::BAR1_BUF;
#include "cl_cuda.h"

/**
 *	\brief	Checking the transmission counter at CUDA device
 *
 *	Key actions:
 *		-# Open CUDA device
 *		-# Open gpumem driver
 *		-# Allocate three buffers in the CUDA memory
 *		-# Mapping buffers in the BAR1 space on CUDA device
 *		-# Filling the buffer 64-bit counter via BAR1
 *		-# Checking buffer in the CUDA device
 *		-# Decimation buffer and transfer to the HOST
 *		-# Transfer result of checking to HOST
 *
 *
 *		Steps 5-8 are carried out in a loop
 *
 *
 */
class TF_TestCnt: public TF_Test {
public:
	TF_TestCnt( int argc, char **argv );
	virtual ~TF_TestCnt();


	virtual int 	Prepare( int cnt );

	virtual void	Start( void );

	virtual void 	Stop( void );

	virtual int		isComplete( void );

	virtual void	StepTable( void );


	static void* ThreadFunc( void* lpvThreadParm );

	void* Execute( void );

	void PrepareInThread( void );

	void CleanupInThread( void );

	void Run( void );

	int	m_argc;
	char** m_argv;

	int	m_isPrepareComplete;
	int	m_isComplete;
	int m_isTerminate;

	int	m_CycleCnt;

	pthread_mutex_t		m_StartMutex;
	pthread_cond_t		m_StartCond;

    pthread_t 			m_hThread;
    pthread_attr_t  	m_attrThread;

    CL_Cuda				*m_pCuda;	//!< Cuda device

    CL_Cuda::BAR1_BUF	m_Bar1[3];	//!< description of buffer in BAR1


    uint64_t			m_CurrentCounter;

    void FillCounter( CL_Cuda::BAR1_BUF *pBar1 );

};

#endif /* TF_TESTCNT_H_ */
