/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <signal.h>


#include <assert.h>


#include "tf_testcnt.h"


static volatile int exit_flag = 0;

void signa_handler(int signo)
{
    exit_flag = 1;
}


//int run_cuda(int argc, char **argv);

int main(int argc, char **argv)
{

	int ret;

    signal(SIGINT, signa_handler);

	try
	{

		TF_TestCnt	*pTest = new TF_TestCnt( argc, argv );

		for( int ii=0; ; ii++)
		{
			if( pTest->Prepare(ii) )
				break;
		}

		pTest->Start();

		for( ; ; )
		{

			if( pTest->isComplete() )
				break;

			if( exit_flag )
			{
				pTest->Stop();
			}

			pTest->StepTable();
		}

		delete pTest; pTest=NULL;

	} catch( ... )
	{

	}

	//fprintf( stderr, "\nPress any key for exit\n" );
	//getchar();

    return ret;
}

