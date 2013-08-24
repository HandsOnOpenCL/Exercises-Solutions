//------------------------------------------------------------------------------
//
// Name:       gameoflife.cl
// 
// Purpose:    Run a naive Conway's game of life - the kernel itself
//
// HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
//
//------------------------------------------------------------------------------

#define ALIVE 1
#define DEAD  0

__kernel void accelerate_life(__global const char* tick, __global char* tock, const unsigned int nx, const unsigned int ny, __local char* block)
{

    // The cell we work on in the loop
    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);

    // Indexes of rows/columns next to idx
    // wrapping around if required
    unsigned int x_l, x_r, y_u, y_d;

    // Calculate indexes
    const unsigned int id = idy * nx + idx;
    x_r = (idx + 1) % nx;
    x_l = (idx == 0) ? nx - 1 : idx - 1;
    y_u = (idy + 1) % ny;
    y_d = (idy == 0) ? ny - 1: idy - 1;

    // Count alive neighbours (out of eight)
    int neighbours = 0;
    if (tick[idy * nx + x_l] == ALIVE) neighbours++;
    if (tick[y_u * nx + x_l] == ALIVE) neighbours++;
    if (tick[y_d * nx + x_l] == ALIVE) neighbours++;
    
    if (tick[idy * nx + x_r] == ALIVE) neighbours++;
    if (tick[y_u * nx + x_r] == ALIVE) neighbours++;
    if (tick[y_d * nx + x_r] == ALIVE) neighbours++;
    
    if (tick[y_u * nx + idx] == ALIVE) neighbours++;
    if (tick[y_d * nx + idx] == ALIVE) neighbours++;

    // Apply game of life rules
    if (tick[id] == ALIVE)
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
