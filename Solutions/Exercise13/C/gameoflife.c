//------------------------------------------------------------------------------
//
// Name:       gameoflife.c
// 
// Purpose:    Run Conway's game of life
//
// HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
//
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define FINALSTATEFILE "final_state.dat"

// Define the state of the cell
#define DEAD  0
#define ALIVE 1

/*************************************************************************************
 * Forward declarations of utility functions
 ************************************************************************************/
void die(const char* message, const int line, const char *file);
void load_board(char* board, const char* file, const unsigned int nx, const unsigned int ny);
void print_board(const char* board, const unsigned int nx, const unsigned int ny);
void save_board(const char* board, const unsigned int nx, const unsigned int ny);
void load_params(const char *file, unsigned int *nx, unsigned int *ny, unsigned int *iterations);

char *getKernelSource(char*);

/*************************************************************************************
 * Main function
 ************************************************************************************/

int main(int argc, char **argv)
{

    // Check we have a starting state file
    if (argc != 5)
    {
        printf("Usage:\n./gameoflife input.dat input.params bx by\n");
        printf("\tinput.dat\tpattern file\n");
        printf("\tinput.params\tparameter file defining board size\n");
        printf("\tbx by\tsizes of thread blocks - must divide the board size equally\n");
        return EXIT_FAILURE;
    }

    // OpenCL setup
    cl_int err;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Find number of platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkError(err, "Finding platforms");
    // Get all platforms
    cl_platform_id platforms[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    checkError(err, "Getting platforms");

    // Secure a device
    for (int i = 0; i < num_platforms; i++)
    {
        err = clGetDeviceIDs(platforms[i], DEVICE, 1, &device, NULL);
        if (err == CL_SUCCESS)
            break;
    }
    if (device == NULL)
        checkError(err, "Getting device");

    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    char *kernel_source = getKernelSource("../gameoflife.cl");
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
        fprintf(stderr, "%s\n", buffer);
        checkError(err, "Building program");
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "accelerate_life", &err);
    checkError(err, "Creating kernel");

    // Board dimensions, work-group sizes and iteration total
    unsigned int nx, ny;
    unsigned int bx = atoi(argv[3]);
    unsigned int by = atoi(argv[4]);
    unsigned int iterations;

    load_params(argv[2], &nx, &ny, &iterations);

    // Allocate memory for boards
    char* h_board = (char *)calloc(nx * ny, sizeof(char));
    if (!h_board)
        die("Could not allocate memory for board", __LINE__, __FILE__);

    cl_mem d_board_tick = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * nx * ny, NULL, &err);
    checkError(err, "Creating buffer d_board_tick");
    
    cl_mem d_board_tock = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * nx * ny, NULL, &err);
    checkError(err, "Creating buffer d_board_tock");

    // Load in the starting state to host board and copy to device
    load_board(h_board, argv[1], nx, ny);
    err = clEnqueueWriteBuffer(queue, d_board_tick, CL_FALSE, 0, sizeof(char) * nx * ny, h_board, 0, NULL, NULL);
    checkError(err, "Writing to buffer d_board_tick");

    // Display the starting state
    printf("Starting state\n");
    print_board(h_board, nx, ny);

    // Set he global and local problem sizes
    const size_t global[2] = {nx, ny};
    const size_t local[2] = {bx, by};

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_board_tick);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_board_tock);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &ny);
    // And allocate the local memory
    err |= clSetKernelArg(kernel, 4, sizeof(char) * (bx + 2) * (by + 2), NULL);
    checkError(err, "Setting kernel args");


    // Loop
    for (unsigned int i = 0; i < iterations; i++)
    {
        // Apply the rules of Life
        err = clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, NULL, NULL);
        checkError(err, "Enqueueing kernel");

        // Swap the boards over
        cl_mem tmp = d_board_tick;
        d_board_tick = d_board_tock;
        d_board_tock = tmp;

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_board_tick);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_board_tock);
        checkError(err, "Setting kernel args");
    }

    // Copy back the memory to the host
    err = clEnqueueReadBuffer(queue, d_board_tick, CL_TRUE, 0, sizeof(char) * nx * ny, h_board, 0, NULL, NULL);
    checkError(err, "Copying from buffer d_board_tick");

    // Display the final state
    printf("Finishing state\n");
    print_board(h_board, nx, ny);

    // Save the final state of the board
    save_board(h_board, nx, ny);

    return EXIT_SUCCESS;
}


/*************************************************************************************
 * Utility functions
 ************************************************************************************/

// Function to load the params file and set up the X and Y dimensions
void load_params(const char* file, unsigned int *nx, unsigned int *ny, unsigned int *iterations)
{
    FILE *fp = fopen(file, "r");
    if (!fp)
        die("Could not open params file.", __LINE__, __FILE__);

    int retval;
    retval = fscanf(fp, "%d\n", nx);
    if (retval != 1)
        die("Could not read params file: nx.", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", ny);
    if (retval != 1)
        die("Could not read params file: ny", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", iterations);
    if (retval != 1)
        die("Could not read params file: iterations", __LINE__, __FILE__);

    fclose(fp);
}

// Function to load in a file which lists the alive cells
// Each line of the file is expected to be: x y 1
void load_board(char* board, const char* file, const unsigned int nx, const unsigned int ny)
{
    FILE *fp = fopen(file, "r");
    if (!fp)
        die("Could not open input file.", __LINE__, __FILE__);

    int retval;
    unsigned int x, y, s;
    while ((retval = fscanf(fp, "%u %u %u\n", &x, &y, &s)) != EOF)
    {
        if (retval != 3)
            die("Expected 3 values per line in input file.", __LINE__, __FILE__);
        if (x > nx - 1)
            die("Input x-coord out of range.", __LINE__, __FILE__);
        if (y > ny - 1)
            die("Input y-coord out of range.", __LINE__, __FILE__);
        if (s != ALIVE)
            die("Alive value should be 1.", __LINE__, __FILE__);

        board[x + y * nx] = ALIVE;
    }

    fclose(fp);
}

// Function to print out the board to stdout
// Alive cells are displayed as O
// Dead cells are displayed as .
void print_board(const char* board, const unsigned int nx, const unsigned int ny)
{
    for (unsigned int i = 0; i < ny; i++)
    {
        for (unsigned int j = 0; j < nx; j++)
        {
            if (board[i * nx + j] == DEAD)
                printf(".");
            else
                printf("O");
        }
        printf("\n");
    }
}

void save_board(const char* board, const unsigned int nx, const unsigned int ny)
{
    FILE *fp = fopen(FINALSTATEFILE, "w");
    if (!fp)
        die("Could not open final state file.", __LINE__, __FILE__);

    for (unsigned int i = 0; i < ny; i++)
    {
        for (unsigned int j = 0; j < nx; j++)
        {
            if (board[i * nx + j] == ALIVE)
                fprintf(fp, "%d %d %d\n", j, i, ALIVE);
        }
    }
}

// Function to display error and exit nicely
void die(const char* message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n",message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

char *getKernelSource(char *filename)
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
