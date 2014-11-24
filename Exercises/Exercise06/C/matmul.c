//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multiplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010 
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Ported to C by Tom Deakin, July 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#include "matmul.h"
#include "matrix_lib.h"
#include "err_code.h"
#include "device_picker.h"

char * kernelsource = "__kernel void mmul(                              \n" \
"   const int N,                                                        \n" \
"   __global float* A,                                                  \n" \
"   __global float* B,                                                  \n" \
"   __global float* C)                                                  \n" \
"{                                                                      \n" \
"}                                                                      \n" \
"\n";

int main(int argc, char *argv[])
{
    float *h_A;             // A matrix
    float *h_B;             // B matrix
    float *h_C;             // C = A*B matrix
    int N;                  // A[N][N], B[N][N], C[N][N]
    int size;               // number of elements in each matrix

    cl_mem d_a, d_b, d_c;   // Matrices in device memory

    double start_time;      // Starting time
    double run_time;        // timing data

    cl_int err;             // error code returned from OpenCL calls
    cl_device_id     device;     // compute device id 
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel;        // compute kernel

    N    = ORDER;
    size = N * N;

    h_A = (float *)malloc(size * sizeof(float));
    h_B = (float *)malloc(size * sizeof(float));
    h_C = (float *)malloc(size * sizeof(float));



//--------------------------------------------------------------------------------
// Create a context, queue and device.
//--------------------------------------------------------------------------------

    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex);

    // Get list of devices
    cl_device_id devices[MAX_DEVICES];
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices)
    {
      printf("Invalid device index (try '--list')\n");
      return EXIT_FAILURE;
    }

    device = devices[deviceIndex];

    char name[MAX_INFO_STRING];
    getDeviceName(device, name);
    printf("\nUsing OpenCL device: %s\n", name);

    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

   // Create a command queue
    commands = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

//--------------------------------------------------------------------------------
// Run sequential version on the host
//--------------------------------------------------------------------------------

    initmat(N, h_A, h_B, h_C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",ORDER);
    for(int i = 0; i < COUNT; i++)
    {
        zero_mat(N, h_C);
        start_time = wtime();

        seq_mat_mul_sdot(N, h_A, h_B, h_C);

        run_time  = wtime() - start_time;
        results(N, h_C, run_time);
    }


//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

    //  Reset A, B and C matrices (just to play it safe)
    initmat(N, h_A, h_B, h_C);

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * size, h_A, &err);
    checkError(err, "Creating buffer d_a");

    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * size, h_B, &err);
    checkError(err, "Creating buffer d_b");

    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                            sizeof(float) * size, NULL, &err);
    checkError(err, "Creating buffer d_c");


//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Naive
//--------------------------------------------------------------------------------

    // Create the comput program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
    checkError(err, "Creating program");

   // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "mmul", &err);
    checkError(err, "Creating kernel");

    printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",N);

    // Do the multiplication COUNT times
    for (int i = 0; i < COUNT; i++)
    {
        zero_mat(N, h_C);

        err =  clSetKernelArg(kernel, 0, sizeof(int),    &N);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_c);
        checkError(err, "Setting kernel arguments");

        start_time = wtime();

        // Execute the kernel over the entire range of C matrix elements ... computing
        // a dot product for each element of the product matrix.  The local work
        // group size is set to NULL ... so I'm telling the OpenCL runtime to
        // figure out a local work group size for me.
        const size_t global[2] = {N, N};
        err = clEnqueueNDRangeKernel(
            commands,
            kernel,
            2, NULL,
            global, NULL,
            0, NULL, NULL);

        checkError(err, "Enqueuing kernel");

        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(float) * size, h_C,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_c");

        results(N, h_C, run_time);

    } // end for loop


//--------------------------------------------------------------------------------
// Clean up!
//--------------------------------------------------------------------------------

    free(h_A);
    free(h_B);
    free(h_C);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}
