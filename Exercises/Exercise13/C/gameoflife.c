//------------------------------------------------------------------------------
//
// Name:       gameoflife.c
// 
// Purpose:    Run a naive Conway's game of life
//
// HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
//
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
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


/*************************************************************************************
 * Game of Life worker method
 ************************************************************************************/

// Apply the rules of life to tick and save in tock
void accelerate_life(const char* tick, char* tock, const int nx, const int ny)
{
    // The cell we work on in the loop
    unsigned int idx;

    // Indexes of rows/columns next to idx
    // wrapping around if required
    unsigned int x_l, x_r, y_u, y_d;

    unsigned int j;
    #pragma omp parallel for private(j, idx, x_l, x_r, y_u, y_d)
    for (unsigned int i = 0; i < ny; i++)
    {
        for (j = 0; j < nx; j++)
        {
            // Calculate indexes
            idx = i * nx + j;
            x_r = (j + 1) % nx;
            x_l = (j == 0) ? nx - 1 : j - 1;
            y_u = (i + 1) % ny;
            y_d = (i == 0) ? ny - 1: i - 1;

            // Count alive neighbours (out of eight)
            int neighbours = 0;
            if (tick[i * nx + x_l] == ALIVE) neighbours++;
            if (tick[y_u * nx + x_l] == ALIVE) neighbours++;
            if (tick[y_d * nx + x_l] == ALIVE) neighbours++;
            
            if (tick[i * nx + x_r] == ALIVE) neighbours++;
            if (tick[y_u * nx + x_r] == ALIVE) neighbours++;
            if (tick[y_d * nx + x_r] == ALIVE) neighbours++;
            
            if (tick[y_u * nx + j] == ALIVE) neighbours++;
            if (tick[y_d * nx + j] == ALIVE) neighbours++;

            // Apply game of life rules
            if (tick[idx] == ALIVE)
            {
                if (neighbours == 2 || neighbours == 3)
                    // Cell lives on
                    tock[idx] = ALIVE;
                else
                    // Cell dies by over/under population
                    tock[idx] = DEAD;
            }
            else
            {
                if (neighbours == 3)
                    // Cell becomes alive through reproduction
                    tock[idx] = ALIVE;
                else
                    // Remains dead
                    tock[idx] = DEAD;
            }

        }
    }
}


/*************************************************************************************
 * Main function
 ************************************************************************************/

int main(int argc, char **argv)
{

    // Check we have a starting state file
    if (argc != 3)
    {
        printf("Usage:\n./gameoflife input.dat input.params\n");
        return EXIT_FAILURE;
    }

    // Board dimensions and iteration total
    unsigned int nx, ny;
    unsigned int iterations;

    load_params(argv[2], &nx, &ny, &iterations);

    // Allocate memory for boards
    char* board_tick = (char *)calloc(nx * ny, sizeof(char));
    char* board_tock = (char *)calloc(nx * ny, sizeof(char));

    if (!board_tick || !board_tock)
        die("Could not allocate memory for board", __LINE__, __FILE__);

    // Load in the starting state to board_tick
    load_board(board_tick, argv[1], nx, ny);

    // Display the starting state
    printf("Starting state\n");
    print_board(board_tick, nx, ny);

    // Loop
    for (unsigned int i = 0; i < iterations; i++)
    {
        // Apply the rules of Life
        accelerate_life(board_tick, board_tock, nx, ny);

        // Swap the boards over
        char *tmp = board_tick;
        board_tick = board_tock;
        board_tock = tmp;
    }

    // Display the final state
    printf("Finishing state\n");
    print_board(board_tick, nx, ny);

    // Save the final state of the board
    save_board(board_tick, nx, ny);

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
