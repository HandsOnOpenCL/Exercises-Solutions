#
# Matrix Multiplication Driver
#
# This is a driver program to test various ways of computing
# the product:
#                 C = A * B
#
# A and B are constant matrices, square and the order is
# set as a constant, ORDER (see definitions.py). This is so
# we can make a quick test of the multiplication result.
#
# History:   C++ version written by Tim Mattson, August 2010 
#            Modified by Simon McIntosh-Smith, September 2011
#            Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
#            Ported to Python by Tom Deakin, July 2013
#

from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time

# A[N][N], B[N][N], C[N][N]
N = ORDER;


# Number of elements in the matrix
size = N * N



# A matrix
h_A = initMat (N, 7)

# B matrix
h_B = initMat (N, 11)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)

## C0 is the product of h_A and h_B for later comparisons:

C0 = numpy.matmul(h_A.reshape([N,N]), h_B.reshape([N,N])).reshape(N*N);

print "\n===== Sequential, matrix mult (dot prod), order", ORDER, "on host CPU ======\n"

for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    print "Skipping as this takes a long time to run!"
    #seq_mat_mul_sdot(N, h_A, h_B, h_C)

    run_time = time() - start_time
    #results(N, h_C, run_time)


# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = initMat (N,  7)
h_B = initMat (N, 11);
h_C = numpy.empty(size).astype(numpy.float32)
C0  = numpy.matmul(h_A.reshape([N,N]), h_B.reshape([N,N])).reshape(N*N);


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive
#--------------------------------------------------------------------------------

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

kernelsource = open("../C_elem.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])
print "\n===== OpenCL, matrix mult, C(i,j) per work item, order", N, "======\n"

# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    mmul(queue, (N, N), None, N, d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, C0, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item
#--------------------------------------------------------------------------------

kernelsource = open("../C_row.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])
print "\n===== OpenCL, matrix mult, C row per work item, order", N, "======\n"
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    mmul(queue, (N,), (ORDER/16,), N, d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, C0, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item, A row in pivate memory
#--------------------------------------------------------------------------------

kernelsource = open("../C_row_priv.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])
print "\n===== OpenCL, matrix mult, C row, A row in priv mem, order", N, "======\n"
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    mmul(queue, (N,), (ORDER/16,), N, d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, C0, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item, A row pivate, B col local
#--------------------------------------------------------------------------------

kernelsource = open("../C_row_priv_bloc.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None])
print "\n===== OpenCL, mat mult, C row, priv A, B cols loc, order", N, "======\n"
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    localmem = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * N)
    mmul(queue, (N,), (ORDER/16,), N,
    	d_a, d_b, d_c, localmem)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, C0, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... blocked
#--------------------------------------------------------------------------------

kernelsource = open("../C_block_form.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])
print "\n==== Parallel matrix mult (blocked), order {0} on device ======\n".format(N)
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    # Work-group computes a block of C. This size is also set
    # in a #define inside the kernel function. Note this blocksize
    # must evenly divide the matrix order
    blocksize = 16

    A_block = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * blocksize * blocksize)
    B_block = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * blocksize * blocksize)
    mmul(queue, (N,N), (blocksize,blocksize), N,
        d_a, d_b, d_c, A_block, B_block)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, C0, run_time)
