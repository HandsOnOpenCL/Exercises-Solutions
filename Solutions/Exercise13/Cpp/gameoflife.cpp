//------------------------------------------------------------------------------
//
// Name:       gameoflife.cpp
// 
// Purpose:    Run a naive Conway's game of life
//
// HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
//
//------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "util.hpp"

//pick up device type from compiler command line or from 
//the default type
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
void die(const std::string message, const int line, const std::string file);
void load_board(std::vector<char>& board, const char* file, const unsigned int nx, const unsigned int ny);
void print_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny);
void save_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny);
void load_params(const char* file, unsigned int *nx, unsigned int *ny, unsigned int *iterations);

#include "err_code.h"

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
    unsigned int bx = atoi(argv[3]);
    unsigned int by = atoi(argv[4]);
    unsigned int iterations;

    load_params(argv[2], &nx, &ny, &iterations);

    // Create OpenCL context, queue and program
    try
    {
        cl::Context context(DEVICE);
        cl::CommandQueue queue(context);
        cl::Program program(context, util::loadProgram("../gameoflife.cl"));
        try
        {
            program.build();
        }
        catch (cl::Error error)
        {
            // If it was a build error then show the error
            if (error.err() == CL_BUILD_PROGRAM_FAILURE)
            {
                std::vector<cl::Device> devices;
                devices = context.getInfo<CL_CONTEXT_DEVICES>();
                std::string built = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
                std::cerr << built << "\n";
            }
            throw error;
        }

        cl::make_kernel
            <cl::Buffer, cl::Buffer, unsigned int, unsigned int, cl::LocalSpaceArg>
            accelerate_life(program, "accelerate_life");

        // Allocate memory for boards
        std::vector<char> h_board(nx * ny);
        cl::Buffer d_board_tick(context, CL_MEM_READ_WRITE, sizeof(char) * nx * ny);
        cl::Buffer d_board_tock(context, CL_MEM_READ_WRITE, sizeof(char) * nx * ny);

        // Load in the starting state to host board and copy to device
        load_board(h_board, argv[1], nx, ny);
        cl::copy(queue, h_board.begin(), h_board.end(), d_board_tick);

        // Display the starting state
        std::cout << "Starting state\n";
        print_board(h_board, nx, ny);

        // Set the global and local problem sizes
        cl::NDRange global(nx, ny);
        cl::NDRange local(bx, by);

        // Allocate local memory
        cl::LocalSpaceArg localmem = cl::Local(sizeof(char) * (bx + 2) * (by + 2));

        // Loop
        for (unsigned int i = 0; i < iterations; i++)
        {
            // Apply the rules of Life
            // Enqueue the kernel
            accelerate_life(cl::EnqueueArgs(queue, global, local), d_board_tick, d_board_tock, nx, ny, localmem);

            // Swap the boards over
            cl::Buffer tmp = d_board_tick;
            d_board_tick = d_board_tock;
            d_board_tock = tmp;
        }

        // Copy back the memory to the host
        cl::copy(queue, d_board_tick, h_board.begin(), h_board.end());

        // Display the final state
        std::cout << "Finishing state\n";
        print_board(h_board, nx, ny);

        // Save the final state of the board
        save_board(h_board, nx, ny);

    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ":\n";
        err_code(err.err());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


/*************************************************************************************
 * Utility functions
 ************************************************************************************/

// Function to load the params file and set up the X and Y dimensions
void load_params(const char* file, unsigned int *nx, unsigned int *ny, unsigned int *iterations)
{
    std::ifstream fp(file);
    if (!fp.is_open())
        die("Could not open params file.", __LINE__, __FILE__);

    int retval;
    fp >> *nx;
    fp >> *ny;
    fp >> *iterations;
    fp.close();
}

// Function to load in a file which lists the alive cells
// Each line of the file is expected to be: x y 1
void load_board(std::vector<char>& board, const char* file, const unsigned int nx, const unsigned int ny)
{
    std::ifstream fp(file);
    if (!fp.is_open())
        die("Could not open input file.", __LINE__, __FILE__);

    int retval;
    unsigned int x, y, s;
    while (fp >> x >> y >> s)
    {
        if (x > nx - 1)
            die("Input x-coord out of range.", __LINE__, __FILE__);
        if (y > ny - 1)
            die("Input y-coord out of range.", __LINE__, __FILE__);
        if (s != ALIVE)
            die("Alive value should be 1.", __LINE__, __FILE__);

        board[x + y * nx] = ALIVE;
    }

    fp.close();
}

// Function to print out the board to stdout
// Alive cells are displayed as O
// Dead cells are displayed as .
void print_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny)
{
    for (unsigned int i = 0; i < ny; i++)
    {
        for (unsigned int j = 0; j < nx; j++)
        {
            if (board[i * nx + j] == DEAD)
                std::cout << ".";
            else
                std::cout << "O";
        }
        std::cout << "\n";
    }
}

void save_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny)
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
void die(const std::string message, const int line, const std::string file)
{
  std::cerr << "Error at line " << line << " of file " << file << ":\n";
  std::cerr << message << "\n";
  exit(EXIT_FAILURE);
}
