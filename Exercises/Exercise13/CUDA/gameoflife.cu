//------------------------------------------------------------------------------
//
// Name:       gameoflife.cu
// 
// Purpose:    CUDA implementation of Conway's game of life
//
// HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
//
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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
void errorCheck(cudaError_t error);

/*************************************************************************************
 * Game of Life worker method - CUDA kernel
 ************************************************************************************/

// Apply the rules of life to tick and save in tock
__global__ void accelerate_life(const char* tick, char* tock, const int nx, const int ny)
{
    // The cell we work on in the loop
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;

    // Index with respect to global array
    unsigned int id = idy * nx + idx;
    unsigned int id_b = (threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1;

    // Copy block to shared memory
    extern __shared__ char block[];
    block[id_b] = tick[id];

    // Copy the halo cells (those around the block) to shared memory
    const unsigned int block_r = (blockIdx.x + 1) % gridDim.x;
    const unsigned int block_l = (blockIdx.x == 0) ? gridDim.x - 1 : blockIdx.x - 1;
    const unsigned int block_u = (blockIdx.y + 1) % gridDim.y;
    const unsigned int block_d = (blockIdx.y  == 0) ? gridDim.y - 1: blockIdx.y - 1;

    // Select the first row of threads
    if (threadIdx.y == 0)
    {
        // Down row
        block[threadIdx.x + 1] = tick[(blockDim.y * block_d + blockDim.y - 1) * nx + idx];
    }
    // Select the last row of threads
    if (threadIdx.y == blockDim.y - 1)
    {
        // Up row
        block[id_b + blockDim.x + 2] = tick[(blockDim.y * block_u) * nx + idx];
    }

    // Select right column of threads
    if (threadIdx.x == blockDim.x - 1)
    {
        // Copy in right
        block[id_b + 1] = tick[nx * idy + (blockDim.x * block_r)];
    }

    // Select left column of threads
    if (threadIdx.x == 0)
    {
        // Copy in left
        block[id_b - 1] = tick[nx * idy + (blockDim.x * block_l + blockDim.x - 1)];
    }


    // Add the 4 corner halo cells
    block[0] = tick[nx * (blockDim.y * block_d + blockDim.y - 1) + (blockDim.x * block_l) + blockDim.x - 1];
    block[blockDim.x + 1] = tick[nx * (blockDim.y * block_d + blockDim.y - 1) + (blockDim.x * block_r)];
    block[(blockDim.x + 2) * (blockDim.y + 1)] = tick[nx * (blockDim.y * block_u) + (blockDim.x * block_l) + blockDim.x - 1];
    block[(blockDim.x + 2) * (blockDim.y + 2) - 1] = tick[nx * (blockDim.y * block_u) + (blockDim.x * block_r)];
    
    __syncthreads();

    // Indexes of rows/columns next to id_b
    unsigned int x_l, x_r, y_u, y_d;

    // Calculate indexes
    x_r = threadIdx.x + 2;
    x_l = threadIdx.x;
    y_u = threadIdx.y + 2;
    y_d = threadIdx.y;

    // Count alive neighbours (out of eight)
    int neighbours = 0;
    if (block[(threadIdx.y + 1) * (blockDim.x + 2) + x_l] == ALIVE) neighbours++;
    if (block[y_u * (blockDim.x + 2) + x_l] == ALIVE) neighbours++;
    if (block[y_d * (blockDim.x + 2) + x_l] == ALIVE) neighbours++;
        
    if (block[(threadIdx.y + 1) * (blockDim.x + 2) + x_r] == ALIVE) neighbours++;
    if (block[y_u * (blockDim.x + 2) + x_r] == ALIVE) neighbours++;
    if (block[y_d * (blockDim.x + 2) + x_r] == ALIVE) neighbours++;
         
    if (block[y_u * (blockDim.x + 2) + threadIdx.x + 1] == ALIVE) neighbours++;
    if (block[y_d * (blockDim.x + 2) + threadIdx.x + 1] == ALIVE) neighbours++;

    // Apply game of life rules
    if (block[id_b] == ALIVE)
    {
        if (neighbours == 2 || neighbours == 3)
            // Cell lives on
            tock[id] = ALIVE;
        else
            // Cell dies by over/under population
            tock[id] = DEAD;
    }
    else
    {
        if (neighbours == 3)
            // Cell becomes alive through reproduction
            tock[id] = ALIVE;
        else
            // Remains dead
            tock[id] = DEAD;
    }

}


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


    // Board dimensions and iteration total
    unsigned int nx, ny;
    unsigned int iterations;
    unsigned int bx = atoi(argv[3]);
    unsigned int by = atoi(argv[4]);

    load_params(argv[2], &nx, &ny, &iterations);

    // Allocate memory for boards
    size_t size = nx * ny * sizeof(char);
    char* h_board = (char *)calloc(nx * ny, sizeof(char));
    char* d_board_tick;
    char* d_board_tock;

    errorCheck(cudaMalloc(&d_board_tick, size));
    errorCheck(cudaMalloc(&d_board_tock, size));

    // Load in the starting state to board_tick
    load_board(h_board, argv[1], nx, ny);

    // Display the starting state
    printf("Starting state\n");
    print_board(h_board, nx, ny);

    // Copy the host array to the device array
    errorCheck(cudaMemcpy(d_board_tick, h_board, size, cudaMemcpyHostToDevice));

    // Define our problem size for CUDA
    dim3 numBlocks(nx/bx, ny/by);
    dim3 numThreads(bx, by);
    size_t sharedMem = sizeof(char) * (bx + 2) * (by + 2);

    // Loop
    for (unsigned int i = 0; i < iterations; i++)
    {
        // Apply the rules of Life
        accelerate_life<<<numBlocks, numThreads, sharedMem>>>(d_board_tick, d_board_tock, nx, ny);
        errorCheck(cudaPeekAtLastError());

        // Swap the boards over
        char *tmp = d_board_tick;
        d_board_tick = d_board_tock;
        d_board_tock = tmp;
    }

    // Copy the device array back to the host
    errorCheck(cudaMemcpy(h_board, d_board_tick, size, cudaMemcpyDeviceToHost));

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
    while ((retval = fscanf(fp, "%d %d %d\n", &x, &y, &s)) != EOF)
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

void errorCheck(cudaError_t error)
{
    if (error != cudaSuccess)
        die(cudaGetErrorString(error), __LINE__, __FILE__);
}

// Function to display error and exit nicely
void die(const char* message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n",message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}
