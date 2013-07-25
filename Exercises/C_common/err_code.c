//------------------------------------------------------------------------------
//
// Name:     err_code()    
//
// Purpose:  Function to output descriptions of errors for an input error code
//
//
// RETURN:   echoes the input error code
//
// HISTORY:  Written by Tim Mattson, June 2010
//
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include "CL/cl.h"
#endif

int err_code (cl_int err_in)
{
    switch (err_in){
	    case CL_INVALID_PLATFORM:
           printf("\n CL_INVALID_PLATFORM\n");
		   break;
	    case CL_INVALID_DEVICE_TYPE:
           printf("\n CL_INVALID_DEVICE_TYPE\n");
		   break;
        case CL_INVALID_CONTEXT:
           printf("\n CL_INVALID_CONTEXT\n");
           break;
        case CL_INVALID_DEVICE:
           printf("\n CL_INVALID_DEVICE\n");
           break;
        case CL_INVALID_VALUE:
           printf("\n CL_INVALID_VALUE\n");
           break;
        case CL_INVALID_QUEUE_PROPERTIES:
           printf("\n CL_INVALID_QUEUE_PROPERTIES\n");
           break;
        case CL_OUT_OF_RESOURCES:
           printf("\n CL_OUT_OF_RESOURCES\n");
           break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
           printf("\n CL_INVALID_PROGRAM_EXECUTABLE\n"); 
           break;
        case CL_INVALID_KERNEL:
           printf("\n CL_INVALID_KERNEL\n"); 
           break;
        case CL_INVALID_KERNEL_ARGS:
           printf("\n CL_INVALID_KERNEL_ARGS\n"); 
           break;
        case CL_INVALID_WORK_DIMENSION:
           printf("\n CL_INVALID_WORK_DIMENSION\n"); 
           break;
        case CL_INVALID_GLOBAL_OFFSET:
           printf("\n CL_INVALID_GLOBAL_OFFSET\n"); 
           break;
        case CL_INVALID_WORK_GROUP_SIZE:
           printf("\n CL_INVALID_WORK_GROUP_SIZE\n"); 
           break;
        case CL_INVALID_WORK_ITEM_SIZE:
           printf("\n CL_INVALID_WORK_ITEM_SIZE\n"); 
           break;
        case CL_INVALID_IMAGE_SIZE:
           printf("\n CL_INVALID_IMAGE_SIZE\n"); 
           break;
        case CL_INVALID_EVENT_WAIT_LIST:
           printf("\n CL_INVALID_EVENT_WAIT_LIST\n"); 
           break;
        case CL_INVALID_MEM_OBJECT:
           printf("\n CL_INVALID_MEM_OBJECT\n"); 
           break;
        case CL_MEM_COPY_OVERLAP:
           printf("\n CL_MEM_COPY_OVERLAP\n"); 
           break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
           printf("\n CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); 
           break;
        case CL_OUT_OF_HOST_MEMORY:
           printf("\n CL_OUT_OF_HOST_MEMORY\n"); 
           break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
           printf("\n CL_PROFILING_INFO_NOT_AVAILABLE\n"); 
           break;
        case CL_INVALID_EVENT:
           printf("\n CL_INVALID_EVENT\n"); 
           break;
        default:
           printf("\n unknown error.\n");
           break;
    }
    return (int)err_in;
}
