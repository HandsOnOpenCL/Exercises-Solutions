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

# A[N][P], B[P][M], C[N][M]
Ndim = ORDER;
Pdim = ORDER;
Mdim = ORDER;

# Number of elements in the matrix
sizeA = Ndim * Pdim
sizeB = Pdim * Mdim
sizeC = Ndim * Mdim


# A matrix
h_A = numpy.empty(sizeA).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(sizeB).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(sizeC).astype(numpy.float32)

print "\n===== Sequential, matrix mult (dot prod), order", ORDER, "on host CPU ======\n"

for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    print "Skipping as this takes a long time to run!"
    #seq_mat_mul_sdot(Mdim, Ndim, Pdim, h_A, h_B, h_C)

    run_time = time() - start_time
    #results(Mdim, Ndim, Pdim, h_C, run_time)


# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(sizeA).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(sizeB).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(sizeC).astype(numpy.float32)


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive
#--------------------------------------------------------------------------------

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

kernelsource = open("../C_elem.cl").read()
program = cl.Program(context, kernelsource).build()

print "\n===== OpenCL, matrix mult, C(i,j) per work item, order", Ndim, "======\n"

# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    program.mmul(queue, (Ndim, Mdim), None, numpy.int32(Mdim), numpy.int32(Ndim), numpy.int32(Pdim), d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(Mdim, Ndim, Pdim, h_C, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item
#--------------------------------------------------------------------------------

kernelsource = open("../C_row.cl").read()
program = cl.Program(context, kernelsource).build()
print "\n===== OpenCL, matrix mult, C row per work item, order", Ndim, "======\n"
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    program.mmul(queue, (Ndim,), (ORDER/16,), numpy.int32(Mdim), numpy.int32(Ndim), numpy.int32(Pdim), d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(Mdim, Ndim, Pdim, h_C, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item, A row in pivate memory
#--------------------------------------------------------------------------------

kernelsource = open("../C_row_priv.cl").read()
program = cl.Program(context, kernelsource).build()
print "\n===== OpenCL, matrix mult, C row, A row in priv mem, order", Ndim, "======\n"
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    program.mmul(queue, (Ndim,), (ORDER/16,), numpy.int32(Mdim), numpy.int32(Ndim), numpy.int32(Pdim), d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(Mdim, Ndim, Pdim, h_C, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item, A row pivate, B col local
#--------------------------------------------------------------------------------

kernelsource = open("../C_row_priv_bloc.cl").read()
program = cl.Program(context, kernelsource).build()
print "\n===== OpenCL, mat mult, C row, priv A, B cols loc, order", Ndim, "======\n"
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    localmem = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * Pdim)
    program.mmul(queue, (Ndim,), (ORDER/16,), numpy.int32(Mdim), numpy.int32(Ndim), numpy.int32(Pdim),
    	d_a, d_b, d_c, localmem)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(Mdim, Ndim, Pdim, h_C, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ...  A and B in block form in local memory
#--------------------------------------------------------------------------------

kernelsource = open("../C_block_form.cl").read()
program = cl.Program(context, kernelsource).build()
print "\n===== OpenCL, A and B in block form in local memory, order", Ndim, "======\n"
blockSize = 16
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    localmem1 = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * blockSize * blockSize)
    localmem2 = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * blockSize * blockSize)
    program.mmul(queue, (Ndim, Mdim), (blockSize, blockSize),
    	numpy.int32(Mdim), numpy.int32(Ndim), numpy.int32(Pdim),
    	d_a, d_b, d_c, localmem1, localmem2)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(Mdim, Ndim, Pdim, h_C, run_time)
