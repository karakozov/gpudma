/*
 * tf_test.h
 *
 *  Created on: Jan 29, 2017
 *      Author: Dmitry Smekhov
 */

#ifndef TF_TEST_H_
#define TF_TEST_H_


/**
 * 	\brief	Base class for testing device
 */
class TF_Test
{

public:

	virtual int 	Prepare( int cnt )=0;

	virtual void	Start( void )=0;

	virtual void 	Stop( void ) {};

	virtual int		isComplete( void ) { return 0; };

	virtual void	StepTable( void ) {};

	virtual void	GetResult( void ) {};
};




#endif /* TF_TEST_H_ */
