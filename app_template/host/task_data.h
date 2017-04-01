
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

struct TaskHostStatus;

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

	void*	ptrCudaIn;			//!< pointer on bar1 buffer in the Cuda memory

	void*	ptrCudaOut;			//!< pointer on output buffer in the Cuda memory

	TaskHostStatus	*ptrHostStatus;		//!< pointer on TaskHostStatus in the Host memory

	unsigned int	indexWr;	//!< block number for next write
	unsigned int	indexRd;	//!< block number for read
	unsigned int	indexMax;	//!< count blocks in output buffer

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
 * 	\brief	Struct for process status in the host memory
 */
struct TaskHostStatus
{

	unsigned int	indexWr;	//!< block number for next write
	unsigned int	indexRd;	//!< block number for read
	//unsigned int	indexMax;	//!< count blocks in output buffer

};

/**
 *  \brief	Struct of data in monitor area in the host memory
 *
 */
struct TaskHostMonitor
{
	TaskHostStatus	status[3];	//!< Status of process

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

	int 	sizeBufferOfKb;		//!< Size buffer [kbytes]. Must be n*64
	int 	countOfCycle;		//!< Number of cycle. 0 - infinitely

	int		sizeBufferOfBytes;	//!< Size of BAR1 buffer in bytes
	int		countOfBuffers;		//!< Conunt of buffers, from 1 to 3

	//void*				decimationBuffers[3];	//!< Buffer in the CUDA memory for


	size_t 	outputSizeBuffer; 	//!< size of output buffer [bytes]
	size_t 	outputSizeBlock;  	//!< size of output block  [bytes]
	size_t 	outputCountBlock; 	//!< count blocks in the output buffer


	uint64_t*	hostBuffer;			//!< data from device

	TaskHostMonitor* hostMonitor;	//!< monitor data in the host memory

	unsigned int hostBlockRd;		//!< count blocks which host received
	unsigned int hostBlockOk;		//!< block without errors
	unsigned int hostBlockError;	//!< block with errors;

	uint64_t	hostExpectData;		//!< expect data for checking

	TaskCheckData	hostCheck;		//!< result of checking host data

	double velosityExtToCudaCurrent;	//!< velosity data transfer from external device to Cuda for last 4 sec
	double velosityExtToCudaAvr;	    //!< velosity data transfer from external device to Cuda from start

	double velosityCudaToHostCurrent;	//!< velosity data transfer from external device to Cuda for last 4 sec
	double velosityCudaToHostAvr;	    //!< velosity data transfer from external device to Cuda from start

	TaskData()
	{
		cycleCnt=0;
		sizeBufferOfBytes=0;
		countOfBuffers=3;
		currentCounter=0;

		hostBuffer=NULL;
		hostMonitor=NULL;

		hostBlockRd=0;
		hostBlockOk=0;
		hostBlockError=0;
		hostExpectData=0;

		velosityExtToCudaCurrent=0;
		velosityExtToCudaAvr=0;
		velosityCudaToHostCurrent=0;
		velosityCudaToHostAvr=0;

	}
};
