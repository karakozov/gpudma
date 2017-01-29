/*
 * TF_TestCnt.h
 *
 *  Created on: Jan 29, 2017
 *      Author: root
 */

#ifndef TF_TESTCNT_H_
#define TF_TESTCNT_H_

#include "tf_test.h"

class TF_TestCnt: public TF_Test {
public:
	TF_TestCnt( int argc, char **argv );
	virtual ~TF_TestCnt();


	virtual int 	Prepare( int cnt );

	virtual void	Start( void );

	virtual void 	Stop( void );

	virtual int		isComplete( void );

	virtual void	StepTable( void );


	int	m_isComplete;

	int	m_CycleCnt;
};

#endif /* TF_TESTCNT_H_ */
