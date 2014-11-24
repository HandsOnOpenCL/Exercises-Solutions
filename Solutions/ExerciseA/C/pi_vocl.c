/*------------------------------------------------------------------------------
 *
 * Name:       pi_vocl.c
 * 
 * Purpose:    Numeric integration to estimate pi
 *
 * HISTORY:    Written by Tim Mattson, May 2010 
 *             Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
 *             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
 *             Updated by Tom Deakin, September 2013
 *             Updated by Tom Deakin, October 2013
*/

#include <stdio.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

double wtime();
char * getKernelSource(char*);



#define INSTEPS (512*512*512)

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: ./pi_vocl num\n");
		printf("\twhere num = 1, 4 or 8\n");
		return EXIT_FAILURE;
	}

	int vector_size = atoi(argv[1]);

    // Define some vector size specific constants
    unsigned int ITERS, WGS;
    if (vector_size == 1)
    {
        ITERS = 262144;
        WGS = 8;
    }
    else if (vector_size == 4)
    {
        ITERS = 262144 / 4;
        WGS = 32;
    }
    else if (vector_size == 8)
    {
        ITERS = 262144 / 8;
        WGS = 64;
    }
    else
    {
        fprintf(stderr, "Invalid vector size\n");
        return EXIT_FAILURE;
    }

    // Set some default values:
    // Default number of steps (updated later to device preferable)
    unsigned int in_nsteps = INSTEPS;
    // Defaultl number of iterations
    unsigned int niters = ITERS;
    unsigned int work_group_size = WGS;

    // Create context, queue and build program
    cl_int err;
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    // Find number of platforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    // Get all platforms
    cl_platform_id platforms[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkError(err, "Getting platforms");
    // Secure a device
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(platforms[i], DEVICE, 1, &device, NULL);
        if (err == CL_SUCCESS)
            break;
    }
    if (device == NULL) checkError(err, "Getting a device");
    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");
    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");
    // Create the compute program from the source buffer
    char *kernel_source = getKernelSource("../pi_vocl.cl");
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);
    checkError(err, "Creating program");
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        checkError(err, "Building program");
    }
    if (vector_size == 1)
    {
        kernel = clCreateKernel(program, "pi", &err);
        checkError(err, "Creating kernel pi");
    }
    else if (vector_size == 4)
    {
        kernel = clCreateKernel(program, "pi_vec4", &err);
        checkError(err, "Creating kernel pi_vec4");
    }
    else if (vector_size == 8)
    {
        kernel = clCreateKernel(program, "pi_vec8", &err);
        checkError(err, "Creating kernel pi_vec8");
    }

    // Now that we know the size of the work_groups, we can set the number of work
    // groups, the actual number of steps, and the step size
    unsigned int nwork_groups = in_nsteps/(work_group_size*niters);

    // Get the max work group size for the kernel pi on our device
    size_t max_size;
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_size), &max_size, NULL);
    checkError(err, "Getting kernel work group size");

    if (max_size > work_group_size)
    {
        work_group_size = max_size;
        nwork_groups = in_nsteps/(nwork_groups*niters);
    }

    if (nwork_groups < 1)
    {
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(nwork_groups), &nwork_groups, NULL);
        checkError(err, "Getting device max compute units");
        work_group_size = in_nsteps/(nwork_groups*niters);
    }

    unsigned int nsteps = work_group_size * niters * nwork_groups;
    float step_size = 1.0f / (float) nsteps;

    // Array to hold partial sum
    float *h_psum = (float*)calloc(nwork_groups, sizeof(float));

    printf("%d work groups of size %d.\n", nwork_groups, work_group_size);
    printf(" %u Integration steps\n", nsteps);

    cl_mem d_partial_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups, NULL, &err);
    checkError(err, "Creating buffer d_partial_sums");

    // Execute the kernel over the entire range of our 1d input data et
    // using the maximum number of work group items for this device
    const size_t global = nwork_groups * work_group_size;
    const size_t local = work_group_size;

    err = clSetKernelArg(kernel, 0, sizeof(int), &niters);
    err |= clSetKernelArg(kernel, 1, sizeof(float), &step_size);
    err |= clSetKernelArg(kernel, 2, sizeof(float) * work_group_size, NULL);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_partial_sums);
    checkError(err, "Setting kernel args");

    // Start the timer
    double rtime = wtime();

    err = clEnqueueNDRangeKernel(
        queue, kernel,
        1, NULL, &global, &local,
        0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clEnqueueReadBuffer(queue, d_partial_sums, CL_TRUE, 0,
        sizeof(float) * nwork_groups, h_psum, 0, NULL, NULL);
    checkError(err, "Reading back d_partial_sums");

    // complete the sum and compute the final integral value on the host
    float pi_res = 0.0f;
    for (unsigned int i = 0; i < nwork_groups; i++)
    {
        pi_res += h_psum[i];
    }
    pi_res *= step_size;

    rtime = wtime() - rtime;

    printf("\nThe calculation ran in %lf seconds\n", rtime);
    printf(" pi = %f for %u steps\n", pi_res, nsteps);

    free(h_psum);
    free(kernel_source);

}

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

