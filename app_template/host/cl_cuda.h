/*
 * cl_cuda.h
 *
 *  Created on: Feb 4, 2017
 *      Author: Dmitry Smekhov
 */

#ifndef CL_CUDA_H_
#define CL_CUDA_H_

#include <stdint.h>
#include <cuda.h>

class CL_Cuda_private;

/**
 * 	\brief	Common actions for CUDA device
 */
class CL_Cuda
{

private:
	CL_Cuda_private	*pd;

public:
	CL_Cuda( int argc, char** argv );
	virtual ~CL_Cuda();

	//! Description buffer in BAR1 space
	struct BAR1_BUF
	{
		int			id;			//!< User id for buffer
		int			state;		//!< Status of buffer
		size_t		sizeOfBytes;//!< Size buffer of bytes
		int 	    page_count;	//!< Count of pages
		int	      	page_size;	//!< Size of page
		void*		cuda_addr;	//!< address in CUDA memory
		uint64_t*   phy_addr;	//!< Array of physical addresses of pages
		void**		app_addr; 	//!< Array of virtual addresses of pages in the application address space

		BAR1_BUF()
		{
			id=-1;
			state=0xA00;
			sizeOfBytes=0;
			page_count=0;
			page_size=0;
			phy_addr=0;
			app_addr=0;
			cuda_addr=0;
		}
	};

	//! Allocate buffer in CUDA memory and map it in BAR1 space
	void AllocateBar1Buffer( int sizeOfKb, BAR1_BUF *pAdr );

	//! Release buffer from BAR1 space and from CUDA memory
	void FreeBar1Buffer( BAR1_BUF *pAdr );

};

#endif /* CL_CUDA_H_ */
