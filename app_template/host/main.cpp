

// System includes
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

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

			usleep( 10000 ); // 100 ms

		}

		//pTest->GetResult();

		delete pTest; pTest=NULL;

	} catch( ... )
	{

	}

	//fprintf( stderr, "\nPress any key for exit\n" );
	//getchar();

    return ret;
}

