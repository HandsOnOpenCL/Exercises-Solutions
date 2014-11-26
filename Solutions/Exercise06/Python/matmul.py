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
#            Modified to assume square matrices by Ben Elgar, November 2014
#

from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time

C_elem_KernelSource = '''
__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C)
{
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0;
    if ((i < N) && (j < N))
    {
        tmp = 0.0f;
        for (k=0; k<N; k++)
        {
            tmp += A[i*N + k] * B[k*N + j];
        }
        C[i*N + j] = tmp;
    }
}
'''

# A[N][N], B[N][N], C[N][N]
N = ORDER;

# Number of elements in the matrix
size = N * N


# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)

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
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

program = cl.Program(context, C_elem_KernelSource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

print "\n===== OpenCL, matrix mult, C(i,j) per work item, order", N, "======\n"

# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    globalrange = (N, N)
    localrange = None

    mmul(queue, globalrange, localrange, N, d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, run_time)
