
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
//           This version automatically produced by genErrCode.py
//           script written by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int err_code (cl_int err_in)
{
    switch (err_in) {

        case CL_SUCCESS :
            printf("\n CL_SUCCESS \n");
            break;
        case CL_DEVICE_NOT_FOUND :
            printf("\n CL_DEVICE_NOT_FOUND \n");
            break;
        case CL_DEVICE_NOT_AVAILABLE :
            printf("\n CL_DEVICE_NOT_AVAILABLE \n");
            break;
        case CL_COMPILER_NOT_AVAILABLE :
            printf("\n CL_COMPILER_NOT_AVAILABLE \n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE :
            printf("\n CL_MEM_OBJECT_ALLOCATION_FAILURE \n");
            break;
        case CL_OUT_OF_RESOURCES :
            printf("\n CL_OUT_OF_RESOURCES \n");
            break;
        case CL_OUT_OF_HOST_MEMORY :
            printf("\n CL_OUT_OF_HOST_MEMORY \n");
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE :
            printf("\n CL_PROFILING_INFO_NOT_AVAILABLE \n");
            break;
        case CL_MEM_COPY_OVERLAP :
            printf("\n CL_MEM_COPY_OVERLAP \n");
            break;
        case CL_IMAGE_FORMAT_MISMATCH :
            printf("\n CL_IMAGE_FORMAT_MISMATCH \n");
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED :
            printf("\n CL_IMAGE_FORMAT_NOT_SUPPORTED \n");
            break;
        case CL_BUILD_PROGRAM_FAILURE :
            printf("\n CL_BUILD_PROGRAM_FAILURE \n");
            break;
        case CL_MAP_FAILURE :
            printf("\n CL_MAP_FAILURE \n");
            break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET :
            printf("\n CL_MISALIGNED_SUB_BUFFER_OFFSET \n");
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
            printf("\n CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST \n");
            break;
        case CL_INVALID_VALUE :
            printf("\n CL_INVALID_VALUE \n");
            break;
        case CL_INVALID_DEVICE_TYPE :
            printf("\n CL_INVALID_DEVICE_TYPE \n");
            break;
        case CL_INVALID_PLATFORM :
            printf("\n CL_INVALID_PLATFORM \n");
            break;
        case CL_INVALID_DEVICE :
            printf("\n CL_INVALID_DEVICE \n");
            break;
        case CL_INVALID_CONTEXT :
            printf("\n CL_INVALID_CONTEXT \n");
            break;
        case CL_INVALID_QUEUE_PROPERTIES :
            printf("\n CL_INVALID_QUEUE_PROPERTIES \n");
            break;
        case CL_INVALID_COMMAND_QUEUE :
            printf("\n CL_INVALID_COMMAND_QUEUE \n");
            break;
        case CL_INVALID_HOST_PTR :
            printf("\n CL_INVALID_HOST_PTR \n");
            break;
        case CL_INVALID_MEM_OBJECT :
            printf("\n CL_INVALID_MEM_OBJECT \n");
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
            printf("\n CL_INVALID_IMAGE_FORMAT_DESCRIPTOR \n");
            break;
        case CL_INVALID_IMAGE_SIZE :
            printf("\n CL_INVALID_IMAGE_SIZE \n");
            break;
        case CL_INVALID_SAMPLER :
            printf("\n CL_INVALID_SAMPLER \n");
            break;
        case CL_INVALID_BINARY :
            printf("\n CL_INVALID_BINARY \n");
            break;
        case CL_INVALID_BUILD_OPTIONS :
            printf("\n CL_INVALID_BUILD_OPTIONS \n");
            break;
        case CL_INVALID_PROGRAM :
            printf("\n CL_INVALID_PROGRAM \n");
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE :
            printf("\n CL_INVALID_PROGRAM_EXECUTABLE \n");
            break;
        case CL_INVALID_KERNEL_NAME :
            printf("\n CL_INVALID_KERNEL_NAME \n");
            break;
        case CL_INVALID_KERNEL_DEFINITION :
            printf("\n CL_INVALID_KERNEL_DEFINITION \n");
            break;
        case CL_INVALID_KERNEL :
            printf("\n CL_INVALID_KERNEL \n");
            break;
        case CL_INVALID_ARG_INDEX :
            printf("\n CL_INVALID_ARG_INDEX \n");
            break;
        case CL_INVALID_ARG_VALUE :
            printf("\n CL_INVALID_ARG_VALUE \n");
            break;
        case CL_INVALID_ARG_SIZE :
            printf("\n CL_INVALID_ARG_SIZE \n");
            break;
        case CL_INVALID_KERNEL_ARGS :
            printf("\n CL_INVALID_KERNEL_ARGS \n");
            break;
        case CL_INVALID_WORK_DIMENSION :
            printf("\n CL_INVALID_WORK_DIMENSION \n");
            break;
        case CL_INVALID_WORK_GROUP_SIZE :
            printf("\n CL_INVALID_WORK_GROUP_SIZE \n");
            break;
        case CL_INVALID_WORK_ITEM_SIZE :
            printf("\n CL_INVALID_WORK_ITEM_SIZE \n");
            break;
        case CL_INVALID_GLOBAL_OFFSET :
            printf("\n CL_INVALID_GLOBAL_OFFSET \n");
            break;
        case CL_INVALID_EVENT_WAIT_LIST :
            printf("\n CL_INVALID_EVENT_WAIT_LIST \n");
            break;
        case CL_INVALID_EVENT :
            printf("\n CL_INVALID_EVENT \n");
            break;
        case CL_INVALID_OPERATION :
            printf("\n CL_INVALID_OPERATION \n");
            break;
        case CL_INVALID_GL_OBJECT :
            printf("\n CL_INVALID_GL_OBJECT \n");
            break;
        case CL_INVALID_BUFFER_SIZE :
            printf("\n CL_INVALID_BUFFER_SIZE \n");
            break;
        case CL_INVALID_MIP_LEVEL :
            printf("\n CL_INVALID_MIP_LEVEL \n");
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE :
            printf("\n CL_INVALID_GLOBAL_WORK_SIZE \n");
            break;
        case CL_INVALID_PROPERTY :
            printf("\n CL_INVALID_PROPERTY \n");
            break;
        default:
            printf("\n unknown error. \n");
            break;

    }
    return (int)err_in;
}

