#------------------------------------------------------------------------------
#
# Name:       gameoflife.py
# 
# Purpose:    Run a naive Conway's game of life
#
# HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
#
#------------------------------------------------------------------------------

import pyopencl as cl
import numpy
import sys

FINALSTATEFILE = "final_state.dat"

# Define the state of the cell
DEAD = 0
ALIVE = 1

#*************************************************************************************
# Main function
#*************************************************************************************

def main():

    # Check we have a starting state file
    if len(sys.argv) != 5:
        print '''Usage:
        python gameoflife.py input.dat input.params bx by
        \tinput.dat\tpattern file
        \tinput.params\tparater file defining board size
        \tbx by\t\tsizes of the thread blocks - must divide the board size equally
        '''
        sys.exit(-1)

    # Board dimensions and iterations total
    (nx, ny, iterations) = load_params(sys.argv[2])
    bx = int(sys.argv[3])
    by = int(sys.argv[4])

    # Create OpenCL context, queue and program
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    with open('../gameoflife.cl', 'r') as f:
        kernelsource = f.read()
    program = cl.Program(context, kernelsource).build()
    accelerate_life = program.accelerate_life
    accelerate_life.set_scalar_arg_dtypes([None, None, numpy.uint32, numpy.uint32, None])

    # Allocate memory for boards
    h_board = numpy.zeros(nx * ny).astype(numpy.int8)
    mf = cl.mem_flags
    d_board_tick = cl.Buffer(context, mf.READ_WRITE, h_board.nbytes)
    d_board_tock = cl.Buffer(context, mf.READ_WRITE, h_board.nbytes)

    # Load in the starting state to host board and copy memory to device
    load_board(h_board, sys.argv[1], nx, ny)
    cl.enqueue_copy(queue, d_board_tick, h_board)

    # Display the starting state
    print 'Starting state'
    print_board(h_board, nx, ny)

    # Set the global and local problem sizes
    global_size = (nx, ny)
    local_size = (bx, by)

    # Allocate local memory
    sizeof_char = numpy.dtype(numpy.int8).itemsize
    localmem = cl.LocalMemory(sizeof_char * (bx + 2) * (by + 2))

    for i in xrange(iterations):
        # Apply the rules of Life
        # Enqueue the kernel
        accelerate_life(queue, global_size, local_size,
            d_board_tick, d_board_tock,
            nx, ny,
            localmem)

        # Swap the boards over
        tmp = d_board_tick
        d_board_tick = d_board_tock
        d_board_tock = tmp

    # Copy back the memory to the host
    cl.enqueue_copy(queue, h_board, d_board_tick)

    # Display the final state
    print 'Final state'
    print_board(h_board, nx, ny)

    # Save the final state of the board
    save_board(h_board, nx, ny)


#*************************************************************************************
# Utility functions
#*************************************************************************************

def load_board(board, data, nx, ny):
    with open(data, 'r') as f:
        for l in f:
            (x, y, s) = map(int, l.split())
            if x < 0 or x > nx -1:
                die("Input x-coord out of range.")
            if y < 0 or y > ny - 1:
                die("Input y-coord out of range.")
            if s != ALIVE:
                die("Alive value should be 1.")
            board[x + y * nx] = ALIVE


def print_board(board, nx, ny):
    for i in xrange(ny):
        for j in xrange(nx):
            if board[i * nx + j] == DEAD:
                sys.stdout.write('.')
            else:
                sys.stdout.write('O')
        sys.stdout.write('\n')
        sys.stdout.flush()


def save_board(board, nx, ny):
    with open(FINALSTATEFILE, 'w') as f:
        for i in xrange(ny):
            for j in xrange(nx):
                if board[i * nx + j] == ALIVE:
                    f.write('{0} {1} {2}\n'.format(j, i, ALIVE))


def load_params(params):
    with open(params, 'r') as f:
        nx = int(f.readline())
        ny = int(f.readline())
        iterations = int(f.readline())

    return (nx, ny, iterations)


def die(message):
    print 'Error:', message
    sys.exit(-1)


if __name__ == "__main__":
    main()
