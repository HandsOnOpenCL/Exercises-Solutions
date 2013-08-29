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

    // Index with respect to global array
    const unsigned int id = idy * nx + idx;

    // Index with respect to local block (work-group size plus a halo border)
    const unsigned int id_b = (get_local_id(1) + 1) * (get_local_size(0) + 2) + get_local_id(0) + 1;

    // Copy block to local memory
    block[id_b] = tick[id];


    // Copy the halo cells (those around the block) to local memory
    const unsigned int block_r = (get_group_id(0) + 1) % get_num_groups(0);
    const unsigned int block_l = (get_group_id(0) == 0) ? get_num_groups(0) - 1 : get_group_id(0) - 1;
    const unsigned int block_u = (get_group_id(1) + 1) % get_num_groups(1);
    const unsigned int block_d = (get_group_id(1) == 0) ? get_num_groups(1) - 1 : get_group_id(1) - 1;

    // Select the first row of work-items
    if (get_local_id(1) == 0)
    {
        // Down row
        block[get_local_id(0) + 1] = tick[(get_local_size(1) * block_d + get_local_size(1) - 1) * nx + idx];
    }
    // Select the last row of work-items
    if (get_local_id(1) == get_local_size(1) - 1)
    {
        // Up row
        block[id_b + get_local_size(0) + 2] = tick[(get_local_size(1) * block_u) * nx + idx];
    }

    // Select the right column of work-items
    if (get_local_id(0) == get_local_size(0) - 1)
    {
        // Copy in right
        block[id_b + 1] = tick[nx * idy + (get_local_size(0) * block_r)];
    }
    // Select the left column of work-items
    if (get_local_id(0) == 0)
    {
        // Copy in left
        block[id_b - 1] = tick[nx * idy + (get_local_size(0) * block_l + get_local_size(0) - 1)];
    }

    // Copy in the 4 corner halo cells
    block[0] = tick[nx * (get_local_size(1) * block_d + get_local_size(1) - 1) + (get_local_size(0) * block_l) + get_local_size(0) - 1];
    block[get_local_size(0) + 1] = tick[nx * (get_local_size(1) * block_d + get_local_size(1) - 1) + (get_local_size(0) * block_r)];
    block[(get_local_size(0) + 2) * (get_local_size(1) + 1)] = tick[nx * (get_local_size(1) * block_u) + (get_local_size(0) * block_l) + get_local_size(0) - 1];
    block[(get_local_size(0) + 2) * (get_local_size(1) + 2) - 1] = tick[nx * (get_local_size(1) * block_u) + (get_local_size(0) * block_r)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Index of the row/columns next to id_b
    unsigned int x_l, x_r, y_u, y_d;

    // Calculate indexes
    x_r = get_local_id(0) + 2;
    x_l = get_local_id(0);
    y_u = get_local_id(1) + 2;
    y_d = get_local_id(1);

    // Count alive neighbours (out of eight)
    int neighbours = 0;
    if (block[(get_local_id(1) + 1) * (get_local_size(0) + 2) + x_l] == ALIVE) neighbours++;
    if (block[y_u * (get_local_size(0) + 2) + x_l] == ALIVE) neighbours++;
    if (block[y_d * (get_local_size(0) + 2) + x_l] == ALIVE) neighbours++;
    
    if (block[(get_local_id(1) + 1) * (get_local_size(0) + 2) + x_r] == ALIVE) neighbours++;
    if (block[y_u * (get_local_size(0) + 2) + x_r] == ALIVE) neighbours++;
    if (block[y_d * (get_local_size(0) + 2) + x_r] == ALIVE) neighbours++;
    
    if (block[y_u * (get_local_size(0) + 2) + get_local_id(0) + 1] == ALIVE) neighbours++;
    if (block[y_d * (get_local_size(0) + 2) + get_local_id(0) + 1] == ALIVE) neighbours++;

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
