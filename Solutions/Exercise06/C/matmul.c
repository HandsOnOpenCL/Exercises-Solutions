//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication driver
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
//
//------------------------------------------------------------------------------

#include "matmul.h"
#include "matrix_lib.h"

char * kernelsource = "__kernel void mmul(                                                    \n" \
"   const int Mdim,                                                     \n" \
"   const int Ndim,                                                     \n" \
"   const int Pdim,                                                     \n" \
"   __global float* A,                                                  \n" \
"   __global float* B,                                                  \n" \
"   __global float* C)                                                  \n" \
"{                                                                      \n" \
"   int k;                                                              \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = get_global_id(1);                                           \n" \
"   float tmp;                                                          \n" \
"   if ( (i < Ndim) && (j <Mdim))                                       \n" \
"   {                                                                   \n" \
"       tmp = 0.0;                                                      \n" \
"       for(k=0;k<Pdim;k++)                                             \n" \
"           tmp += A[i*Ndim+k] * B[k*Pdim+j];                           \n" \
"       C[i*Ndim+j] = tmp;                                              \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
"\n";

int main(void)
{
    float *h_A;             // A matrix
    float *h_B;             // B matrix
    float *h_C;             // C = A*B matrix
    int Mdim, Ndim, Pdim;   // A[N][P], B[P][M], C[N][M]
    int szA, szB, szC;      // number of elements in each matrix

    cl_mem d_a, d_b, d_c;   // Matrices in device memory

    double start_time;      // Starting time
    double run_time;        // timing data

    cl_int err;             // error code returned from OpenCL calls
    cl_device_id     device_id;     // compute device id 
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel;        // compute kernel

    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    szA = Ndim * Pdim;
    szB = Pdim * Mdim;
    szC = Ndim * Mdim;

    h_A = (float *)malloc(szA * sizeof(float));
    h_B = (float *)malloc(szB * sizeof(float));
    h_C = (float *)malloc(szC * sizeof(float));

    initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",ORDER);
    for(int i = 0; i < COUNT; i++)
    {
        zero_mat(Ndim, Mdim, h_C);
        start_time = wtime();

        seq_mat_mul_sdot(Mdim, Ndim, Pdim, h_A, h_B, h_C);

        run_time  = wtime() - start_time;
        results(Mdim, Ndim, Pdim, h_C, run_time);
    }

//--------------------------------------------------------------------------------
// Create a context, queue and device.
//--------------------------------------------------------------------------------

    // Set up OpenCL context. queue, kernel, etc.
    cl_uint numPlatforms;
    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to find a platform!\n",err_code(err));
        return EXIT_FAILURE;
    }
    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to get the platform!\n",err_code(err));
        return EXIT_FAILURE;
    }
    // Secure a device
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
            break;
    }
    if (device_id == NULL)
    {
        printf("Error: Failed to create a device group!\n",err_code(err));
        return EXIT_FAILURE;
    }

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

    //  Reset A, B and C matrices (just to play it safe)
    initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * szA, h_A, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: failed to create buffer\n", err_code(err));
        return EXIT_FAILURE;
    } 
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * szB, h_B, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: failed to create buffer\n", err_code(err));
        return EXIT_FAILURE;
    }
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                            sizeof(float) * szC, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: failed to create buffer\n", err_code(err));
        return EXIT_FAILURE;
    }


//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Naive
//--------------------------------------------------------------------------------

    // Create the comput program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: could not create program\n", err_code(err));
        return EXIT_FAILURE;
    }
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "mmul", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",Ndim);

    // Do the multiplication COUNT times
    for (int i = 0; i < COUNT; i++)
    {
        zero_mat(Ndim, Mdim, h_C);

        err =  clSetKernelArg(kernel, 0, sizeof(int),    &Mdim);
        err |= clSetKernelArg(kernel, 1, sizeof(int),    &Ndim);
        err |= clSetKernelArg(kernel, 2, sizeof(int),    &Pdim);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_c);

        if (err != CL_SUCCESS)
        {
            printf("Error: Could not set kernel arguments %d\n", err);
            return EXIT_FAILURE;
        }

        start_time = wtime();

        // Execute the kernel over the entire range of C matrix elements ... computing
        // a dot product for each element of the product matrix.  The local work
        // group size is set to NULL ... so I'm telling the OpenCL runtime to
        // figure out a local work group size for me.
        size_t global[2] = {Ndim, Mdim};
        err = clEnqueueNDRangeKernel(
            commands,
            kernel,
            1, NULL,
            global, NULL,
            0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to execute kernel\n", err_code(err));
            return EXIT_FAILURE;
        }

        err = clFinish(commands);
        if (err != CL_SUCCESS)
        {
            printf("Error: waiting for queue to finish failed\n", err_code(err));
            return EXIT_FAILURE;
        }

        run_time = wtime() - start_time;

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(float) * szC, h_C,
            0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read buffer\n", err_code(err));
            return EXIT_FAILURE;
        }

        results(Mdim, Ndim, Pdim, h_C, run_time);

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
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}
