//------------------------------------------------------------------------------
//
// Name:       vadd_three.c
//
// Purpose:    Elementwise addition of D = A + B + C
//
// HISTORY:    Initial version based on vadd.c, written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Ported back to C by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern int output_device_info(cl_device_id );

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, c and d

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum d = a+b+c
//
// input: a, b and c float vectors of length count
//
// output: d float vector of length count holding the sum a + b + c
//

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   __global float* d,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       d[i] = a[i] + b[i] + c[i];                                             \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    cl_int          err;               // error code returned from OpenCL calls

    size_t dataSize = sizeof(float) * LENGTH;
    float*       h_a = (float *)malloc(dataSize);       // a vector
    float*       h_b = (float *)malloc(dataSize);       // b vector
    float*       h_c = (float *)malloc(dataSize);       // c vector
    float*       h_d = (float *)malloc(dataSize);       // d vector (result)
    unsigned int correct;           // number of correct results

    size_t global;                  // global domain size

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel

    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the input c vector
    cl_mem d_d;                     // device memory used for the output d vector

    // Fill vectors a and b with random float values
    int i = 0;
    for(i = 0; i < LENGTH; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_c[i] = rand() / (float)RAND_MAX;
    }

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Getting device");

    err = output_device_info(device_id);
    checkError(err, "Outputting device info");

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    ko_vadd = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b, e, g) arrays in device memory
    // NB: we copy the host pointers here too
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  dataSize, h_a, &err);
    checkError(err, "Creating buffer d_a");
    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  dataSize, h_b, &err);
    checkError(err, "Creating buffer d_b");
    d_c  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  dataSize, h_c, &err);
    checkError(err, "Creating buffer d_c");

    // Create the output arrays in device memory
    d_d  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, dataSize, NULL, &err);
    checkError(err, "Creating buffer d_d");

    const int count = LENGTH;

    // Enqueue kernel
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(cl_mem), &d_d);
    err |= clSetKernelArg(ko_vadd, 4, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel 1st time");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer( commands, d_d, CL_TRUE, 0, sizeof(float) * count, h_d, 0, NULL, NULL );
    checkError(err, "Reading back d_f");

    // Test the results
    correct = 0;
    float tmp;

    for(i = 0; i < count; i++)
    {
        tmp = h_a[i] + h_b[i] + h_c[i];     // assign element i of a+b+c to tmp
        tmp -= h_d[i];                               // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)                        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf(" tmp %f h_a %f h_b %f h_c %f h_d %f\n",tmp, h_a[i], h_b[i], h_c[i], h_d[i]);
        }
    }

    // summarize results
    printf("D = A+B+C:  %d out of %d results were correct.\n", correct, count);

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);

    return 0;
}

