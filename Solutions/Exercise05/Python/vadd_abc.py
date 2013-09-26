#
# Vadd
#
# Element wise addition of three vectors at a time (R=A+B+C)
# Asks the user to select a device at runtime
#
# History: Initial version based on vadd.c, written by Tim Mattson, June 2011
#          Ported to C++ Wrapper API by Benedict Gaster, September 2011
#          Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#

# Import the Python OpenCL API
import pyopencl as cl
# Import the Python Maths Library (for vectors)
import numpy

#------------------------------------------------------------------------------

# tolerance used in floating point comparisons
TOL = 0.001
# length of vectors a, b and c
LENGTH = 1024

#------------------------------------------------------------------------------
#
# Kernel: vadd
#
# To compute the elementwise sum r = a + b + c
#
# Input: a, b and c float vectors of length count
# Output r float vector of length count holding the sum a + b + cs

kernelsource = """
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    __global float* r,
    const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
        r[i] = a[i] + b[i] + c[i];
}
"""

#------------------------------------------------------------------------------

# Main procedure

# Create a compute context
# Ask the user to select a platform/device on the CLI
context = cl.create_some_context()

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer
# and build it
program = cl.Program(context, kernelsource).build()

# Create a, b and c vectors and fill with random float values
# Create empty vectors for r
h_a = numpy.random.rand(LENGTH).astype(numpy.float32)
h_b = numpy.random.rand(LENGTH).astype(numpy.float32)
h_c = numpy.random.rand(LENGTH).astype(numpy.float32)
h_r = numpy.empty(LENGTH).astype(numpy.float32)

# Create the input (a, b, c) arrays in device memory and copy data from host
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)
# Create the output (r) array in device memory
d_r = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_r.nbytes)

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, None, numpy.uint32])
vadd(queue, h_a.shape, None, d_a, d_b, d_c, d_r, LENGTH)

# Read back the results from the compute device
cl.enqueue_copy(queue, h_r, d_r)

# Test the results
correct = 0;
for a, b, c, r in zip(h_a, h_b, h_c, h_r):
    tmp = a + b + c
    # compute the deviation of expected and output result
    tmp -= r
    # correct if square deviation is less than tolerance squared
    if tmp*tmp < TOL*TOL:
        correct += 1
    else:
        print "tmp", tmp, "h_a", a, "h_b", b, "h_c", c, "h_r", r

# Summarize results
print "1 vector adds to find R = A+B+C:", correct, "out of", LENGTH, "results were correct."
