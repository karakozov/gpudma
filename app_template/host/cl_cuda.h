/*
 * cl_cuda.h
 *
 *  Created on: Feb 4, 2017
 *      Author: user52
 */

#ifndef CL_CUDA_H_
#define CL_CUDA_H_


class CL_Cuda_prived;

/**
 * 	\brief	Common actions for CUDA device
 */
class CL_Cuda
{

	CL_Cuda_prived	*pd;

public:
	CL_Cuda( int argc, char** argv );
	virtual ~CL_Cuda();
};

#endif /* CL_CUDA_H_ */
