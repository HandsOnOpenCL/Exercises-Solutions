/*------------------------------------------------------------------------------
 *
 * Name:       pi_ocl.c
 * 
 * Purpose:    Numeric integration to estimate pi
 *
 * HISTORY:    Written by Tim Mattson, May 2010 
 *             Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
 *             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
 *             Ported back to C by Tom Deakin, July 2013
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


extern double wtime();       // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id );
char* err_code (cl_int);

//------------------------------------------------------------------------------
char * getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);

    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}


//------------------------------------------------------------------------------

#define INSTEPS (512*512*512)
#define ITERS (262144)

//------------------------------------------------------------------------------

int main(void)
{
    float *h_psum;              // vector to hold partial sum
    int in_nsteps = INSTEPS;    // default number of steps (updated later to device preferable)
    int niters = ITERS;         // number of iterations
    int nsteps;
    float step_size;
    size_t nwork_groups;
    size_t max_size, work_group_size = 8;
    float pi_res;

    cl_mem d_partial_sums;

    char *kernelsource = getKernelSource("../pi_ocl.cl");             // Kernel source

    cl_int err;
    cl_device_id     device_id;     // compute device id 
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel_pi;     // compute kernel

    // Set up OpenCL context. queue, kernel, etc.
    cl_uint numPlatforms;
    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to find a platform!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }
    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to get the platform!\n%s\n",err_code(err));
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
        printf("Error: Failed to create a device group!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }
    // Output information
    err = output_device_info(device_id);
    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }
    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }
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
    kernel_pi = clCreateKernel(program, "pi", &err);
    if (!kernel_pi || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Find kernel work-group size
    err = clGetKernelWorkGroupInfo (kernel_pi, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get kernel work-group info\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }
    // Now that we know the size of the work-groups, we can set the number of
    // work-groups, the actual number of steps, and the step size
    nwork_groups = in_nsteps/(work_group_size*niters);

    if (nwork_groups < 1)
    {
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &nwork_groups, NULL);
        work_group_size = in_nsteps / (nwork_groups * niters);
    }

    nsteps = work_group_size * niters * nwork_groups;
    step_size = 1.0f/(float)nsteps;
    h_psum = calloc(sizeof(float), nwork_groups);
    if (!h_psum)
    {
        printf("Error: could not allocate host memory for h_psum\n");
        return EXIT_FAILURE;
    }

    printf(" %ld work-groups of size %ld. %d Integration steps\n",
            nwork_groups,
            work_group_size,
            nsteps);

    d_partial_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create buffer\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Set kernel arguments
    err  = clSetKernelArg(kernel_pi, 0, sizeof(int), &niters);
    err |= clSetKernelArg(kernel_pi, 1, sizeof(float), &step_size);
    err |= clSetKernelArg(kernel_pi, 2, sizeof(float) * work_group_size, NULL);
    err |= clSetKernelArg(kernel_pi, 3, sizeof(cl_mem), &d_partial_sums);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments!\n");
        return EXIT_FAILURE;
    }

    // Execute the kernel over the entire range of our 1D input data set
    // using the maximum number of work items for this device
    size_t global = nwork_groups * work_group_size;
    size_t local = work_group_size;
    double rtime = wtime();
    err = clEnqueueNDRangeKernel(
        commands,
        kernel_pi,
        1, NULL,
        &global,
        &local,
        0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }


    err = clEnqueueReadBuffer(
        commands,
        d_partial_sums,
        CL_TRUE,
        0,
        sizeof(float) * nwork_groups,
        h_psum,
        0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read buffer\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // complete the sum and compute the final integral value on the host
    pi_res = 0.0f;
    for (unsigned int i = 0; i < nwork_groups; i++)
    {
        pi_res += h_psum[i];
    }
    pi_res *= step_size;

    rtime = wtime() - rtime;

    printf("\nThe calculation ran in %lf seconds\n", rtime);
    printf(" pi = %f for %d steps\n", pi_res, nsteps);

    // clean up
    clReleaseMemObject(d_partial_sums);
    clReleaseProgram(program);
    clReleaseKernel(kernel_pi);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(kernelsource);
    free(h_psum);
}
