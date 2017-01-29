/*
 * TF_TestCnt.cpp
 *
 *  Created on: Jan 29, 2017
 *      Author: root
 */

#include "tf_testcnt.h"

#include "stdio.h"

int init_cuda(int argc, char **argv);
int run_cuda( void );

TF_TestCnt::TF_TestCnt( int argc, char **argv )
{
	// TODO Auto-generated constructor stub

	m_isComplete=0;
	m_CycleCnt=0;

	init_cuda( argc, argv );
}

TF_TestCnt::~TF_TestCnt() {
	// TODO Auto-generated destructor stub
}

int 	TF_TestCnt::Prepare( int cnt )
{
		if( cnt )
			return 1;

		return 1;
}

void	TF_TestCnt::Start( void )
{

}

void 	TF_TestCnt::Stop( void )
{
	m_isComplete=1;
}

int		TF_TestCnt::isComplete( void )
{
		return m_isComplete;
}

void	TF_TestCnt::StepTable( void )
{

	run_cuda();

	m_CycleCnt++;
	//printf( "  %10d \r", m_CycleCnt );
}
