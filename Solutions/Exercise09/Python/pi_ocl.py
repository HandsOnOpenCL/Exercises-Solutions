#
# Pi reduction
#
# Numeric integration to estimate pi
# Asks the user to select a device at runtime
#
# History: C version written by Tim Mattson, May 2010
#          Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
#          C++ version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#


import pyopencl as cl
import numpy
from time import time

# Some constant values
INSTEPS = 512*512*512
ITERS = 262144

# Set some default values:
# Default number of steps (updated later to device prefereable)
in_nsteps = INSTEPS
# Default number of iterations
niters = ITERS

# Create context, queue and build program
context = cl.create_some_context()
queue = cl.CommandQueue(context)
kernelsource = open("../pi_ocl.cl").read()
program = cl.Program(context, kernelsource).build()
pi = program.pi
pi.set_scalar_arg_dtypes([numpy.int32, numpy.float32, None, None])

# Get the max work group size for the kernel pi on our device
device = context.devices[0]
work_group_size = program.pi.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)


# Now that we know the size of the work_groups, we can set the number of work
# groups, the actual number of steps, and the step size
nwork_groups = in_nsteps/(work_group_size*niters)

if nwork_groups < 1:
	nwork_groups = device.max_compute_units
	work_group_size = in_nsteps/(nwork_groups*niters)

nsteps = work_group_size * niters * nwork_groups
step_size = 1.0 / float(nsteps)

# vector to hold partial sum
h_psum = numpy.empty(nwork_groups).astype(numpy.float32)

print nwork_groups, "work groups of size", work_group_size, ".",
print nsteps, "Integration steps"

d_partial_sums = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_psum.nbytes)

# Start the timer
rtime = time()

# Execute the kernel over the entire range of our 1d input data et
# using the maximum number of work group items for this device
# Set the global and local size as tuples
global_size = ((nwork_groups * work_group_size),)
local_size = ((work_group_size),)
localmem = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * work_group_size)

pi(queue, global_size, local_size,
	niters, step_size,
	localmem, d_partial_sums)

cl.enqueue_copy(queue, h_psum, d_partial_sums)

# complete the sum and compute the final integral value
pi_res = h_psum.sum() * step_size

# Stop the timer
rtime = time() - rtime
print "The calculation ran in", rtime, "seconds"
print "pi =", pi_res, "for", nsteps, "steps"

