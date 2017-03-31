/*
 * TF_TestCnt.h
 *
 *  Created on: Jan 29, 2017
 *      Author: Dmitry Smekhov
 */

#ifndef TF_TESTCNT_H_
#define TF_TESTCNT_H_


#include "tf_testthread.h"

//class CL_Cuda;
//struct CL_Cuda::BAR1_BUF;
#include "cl_cuda.h"


struct TaskData;
struct TaskBufferStatus;

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


	virtual void StepTable( void );

	virtual void PrepareInThread( void );

	virtual void CleanupInThread( void );

	virtual void Run( void );

	virtual void GetResult( void );

	//! Number of arguments
	int	m_argc;

	//! Pointers to arguments
	char** m_argv;


	int m_SizeBufferOfKb;	//!< Size buffer [kbytes]. Must be n*64

	int m_CountOfCycle;		//!< Number of cycle. 0 - infinitely


	struct TaskData		*td;		//!< Local data for test

    CL_Cuda				*m_pCuda;	//!< Cuda device


    //! Fill buffer in Cuda memory via BAR1
    void FillCounter( CL_Cuda::BAR1_BUF *pBar1 );

    //! Print results for buffer
    void GetResultBuffer( int nbuf );

};

#endif /* TF_TESTCNT_H_ */
