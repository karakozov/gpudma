
#include "cl_cuda.h"

//! Number of task for check one buffer
const int TaskCounts=32;


/**
 * 	\brief	Struct for check data in one task for one buffer
 */
struct TaskCheckData
{
	unsigned int	flagError;		//!< 1 - error in current runs
	unsigned int	cntError;		//!< number of errors for all runs

	unsigned int	nblock[16];			//!< number block
	unsigned int	adr[16];			//!< address into block
	uint64_t		expect_data[16];	//!< expect data
	uint64_t		receive_data[16];	//!< receive data

	TaskCheckData()
	{
		for( int ii=0; ii<16; ii++ )
		{
			nblock[ii]=0;
			adr[ii]=0;
			expect_data[ii]=0;
			receive_data[ii]=0;
		}
		flagError=0;
		cntError=0;
	}


};

/**
 * 	\brief	Struct for status calculate
 */
struct TaskBufferStatus
{
	unsigned int irqFlag;		//!< 1 - ready data in bar1 buffer
	unsigned int res0;
	unsigned int res1;
	unsigned int blockRd;		//!< count of read buffer
	unsigned int blockOk;		//!< count of correct buffers
	unsigned int blockError;	//!< count of buffer with errors
	unsigned int sizeOfKBytes;	//!< size of buffers in kilobytes

	void*	ptrCudaIn;	//!< pointer on bar1 buffer in the Cuda memory
	void*	ptrCudaOut;	//!< pointer on output buffer in the Cuda memory

	TaskCheckData	check[ TaskCounts ]; //!< current results for test one buffer
};


/**
 *  \brief	Struct of data in monitor area in BAR1
 *
 */
struct TaskMonitor
{
	TaskBufferStatus	block[3];	//!< Status of buffer0
	int		sig;					//!< signature: 0xAA24
	int		flagExit;				//!< 1 - exit from programm
	int		res0;
	int		res1;

};


/**
 * 	\brief	collection data for TF_TestCnt
 */
struct TaskData
{
	TaskMonitor*		ptrMonitor;	//!< address monitor struct in the HOST memory
	CL_Cuda::BAR1_BUF	monitor;	//!< description of monitor buffer in BAR1
	CL_Cuda::BAR1_BUF	bar1[3];	//!< description of buffer in BAR1

	uint64_t			currentCounter;	//!< Current value for fill buffers

	int					cycleCnt;

	int					sizeBufferOfBytes;	//!< Size of BAR1 buffer in bytes
	int					countOfBuffers;		//!< Conunt of buffers, from 1 to 3

	void*				decimationBuffers[3];	//!< Buffer in the CUDA memory for


	TaskData()
	{
		cycleCnt=0;
		sizeBufferOfBytes=0;
		countOfBuffers=0;
		currentCounter=0;
	}
};
