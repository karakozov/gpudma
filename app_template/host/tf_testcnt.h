/*
 * TF_TestCnt.h
 *
 *  Created on: Jan 29, 2017
 *      Author: root
 */

#ifndef TF_TESTCNT_H_
#define TF_TESTCNT_H_


#include "tf_testthread.h"

//class CL_Cuda;
//struct CL_Cuda::BAR1_BUF;
#include "cl_cuda.h"


struct TaskData;

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
class TF_TestCnt: public TF_TestThread
{
public:
	TF_TestCnt( int argc, char **argv );
	virtual ~TF_TestCnt();


	virtual void	StepTable( void );

	virtual void PrepareInThread( void );

	virtual void CleanupInThread( void );

	virtual void Run( void );

	int	m_argc;
	char** m_argv;





	struct TaskData		*td;		//!< Local data for test

    CL_Cuda				*m_pCuda;	//!< Cuda device

    void FillCounter( CL_Cuda::BAR1_BUF *pBar1 );


};

#endif /* TF_TESTCNT_H_ */
