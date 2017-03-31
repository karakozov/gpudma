/*
 * TF_TestThread.h
 *
 *  Created on: Jan 29, 2017
 *      Author: Dmitry Smekhov
 */

#ifndef TF_TestThread_H_
#define TF_TestThread_H_

#include <pthread.h>
#include "tf_test.h"


/**
 *	\brief	Base class for application with thread
 *
 *
 *
 */
class TF_TestThread: public TF_Test {
public:
	TF_TestThread( int argc, char **argv );
	virtual ~TF_TestThread();


	virtual int 	Prepare( int cnt );

	virtual void	Start( void );

	virtual void 	Stop( void );

	virtual int		isComplete( void );

	virtual void	StepTable( void ) {};


	static void* ThreadFunc( void* lpvThreadParm );

	void* Execute( void );

	virtual void PrepareInThread( void ) {};

	virtual void CleanupInThread( void ) {};

	virtual void Run( void ) {};


	int	m_isPrepareComplete;
	int	m_isComplete;
	int m_isTerminate;

	int	m_CycleCnt;

	pthread_mutex_t		m_StartMutex;
	pthread_cond_t		m_StartCond;

    pthread_t 			m_hThread;
    pthread_attr_t  	m_attrThread;

    int GetFromCommnadLine( int argc, char **argv, char* name, int defValue );

};

#endif /* TF_TestThread_H_ */
