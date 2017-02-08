
#include "cl_cuda.h"

/**
 * 	\brief	Struct for status calculate
 */
struct TaskBufferStatus
{
	int		irqFlag;	//!< 1 - ready data in bar1 buffer
	int		res0;
	int		blockRd;	//!< count of read buufer
	int		res1;
	void*	ptrCudaIn;	//!< pointer on bar1 buffer in the Cuda memory
	void*	ptrCudaOut;	//!< pointer on output buffer in the Cuda memory
};


/**
 *  \brief	Struct of data in monitor area in BAR1
 *
 */
struct TaskMonitor
{
	TaskBufferStatus	block0;		//!< Status of buffer0
	TaskBufferStatus	block1;		//!< Status of buffer1
	TaskBufferStatus	block2;		//!< Status of buffer2
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
	}
};
