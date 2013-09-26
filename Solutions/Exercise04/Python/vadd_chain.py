#
# Vadd
#
# Element wise addition of two vectors at a time in a chain (C=A+B; D=C+E; F=D+G)
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
# To compute the elementwise sum c = a + b
#
# Input: a and b float vectors of length count
# Output c float vector of length count holding the sum a + b

kernelsource = """
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
        c[i] = a[i] + b[i];
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

# Create a, b, e and g vectors and fill with random float values
# Create empty vectors for c, d and f
h_a = numpy.random.rand(LENGTH).astype(numpy.float32)
h_b = numpy.random.rand(LENGTH).astype(numpy.float32)
h_c = numpy.empty(LENGTH).astype(numpy.float32)
h_d = numpy.empty(LENGTH).astype(numpy.float32)
h_e = numpy.random.rand(LENGTH).astype(numpy.float32)
h_f = numpy.empty(LENGTH).astype(numpy.float32)
h_g = numpy.random.rand(LENGTH).astype(numpy.float32)

# Create the input (a, b, e, g) arrays in device memory and copy data from host
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_e = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_e)
d_g = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_g)
# Create the output (c, d, f) array in device memory
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)
d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)
d_f = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_f.nbytes)

vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
vadd(queue, h_a.shape, None, d_a, d_b, d_c, LENGTH)

# Enqueue the kernel again, but with different arguments
vadd(queue, h_e.shape, None, d_e, d_c, d_d, LENGTH)

# Enqueue the kernel a third time, again with different arguments
vadd(queue, h_g.shape, None, d_g, d_d, d_f, LENGTH)


# Read back the results from the compute device
cl.enqueue_copy(queue, h_f, d_f)

# Test the results
correct = 0;
for a, b, e, f, g in zip(h_a, h_b, h_e, h_f, h_g):
    tmp = a + b + e + g
    # compute the deviation of expected and output result
    tmp -= f
    # correct if square deviation is less than tolerance squared
    if tmp*tmp < TOL*TOL:
        correct += 1
    else:
        print "tmp", tmp, "h_a", a, "h_b", b, "h_e", e, "h_g", g, "h_f", f

# Summarize results
print "3 vector adds to find F = A+B+E+G:", correct, "out of", LENGTH, "results were correct."
