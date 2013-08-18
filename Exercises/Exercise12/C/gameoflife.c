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

// Define the board size
#define NX (6)
#define NY (6)

#define DEAD  0
#define ALIVE 1

void print_board(char* board)
{
    for (unsigned int i = 0; i < NY; i++)
    {
        for (unsigned int j = 0; j < NX; j++)
        {
            if (board[i * NX + j] == DEAD)
                printf(".");
            else
                printf("O");
        }
        printf("\n");
    }
}

// Apply the rules of life to tick and save in tock
void accelerate_life(char* tick, char* tock)
{
    // The cell we work on in the loop
    unsigned int idx;

    // Indexes of rows/columns next to idx
    // wrapping around if required
    unsigned int x_l, x_r, y_u, y_d;

    for (unsigned int i = 0; i < NY; i++)
    {
        for (unsigned int j = 0; j < NX; j++)
        {
            // Calculate indexes
            idx = i * NX + j;
            x_r = (j + 1) % NX;
            x_l = (j == 0) ? NY - 1 : j - 1;
            y_u = (i + 1) % NX;
            y_d = (i == 0) ? NX - 1: i - 1;

            // Count alive neighbours (out of eight)
            int neighbours = 0;
            if (tick[i * NX + x_l] == ALIVE) neighbours++;
            if (tick[y_u * NX + x_l] == ALIVE) neighbours++;
            if (tick[y_d * NX + x_l] == ALIVE) neighbours++;
            
            if (tick[i * NX + x_r] == ALIVE) neighbours++;
            if (tick[y_u * NX + x_r] == ALIVE) neighbours++;
            if (tick[y_d * NX + x_r] == ALIVE) neighbours++;
            
            if (tick[y_u * NX + j] == ALIVE) neighbours++;
            if (tick[y_d * NX + j] == ALIVE) neighbours++;

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

int main(void)
{
    // Arrays for boards
    //char* board_tick = (char *)calloc(NX * NY, sizeof(char));
    char board_tick[] = {0,0,0,0,0,0,
                         0,1,1,0,0,0,
                         0,1,1,0,0,0,
                         0,0,0,1,1,0,
                         0,0,0,1,1,0,
                         0,0,0,0,0,0};

    char* board_tock = (char *)calloc(NX * NY, sizeof(char));

    if (!board_tick || !board_tock)
    {
        printf("Error! Could not allocate memory for board\n");
        return EXIT_FAILURE;
    }

    printf("Starting state\n");
    print_board(board_tick);

    accelerate_life(board_tick, board_tock);

    printf("Finishing state\n");
    print_board(board_tock);

    return EXIT_SUCCESS;
}
