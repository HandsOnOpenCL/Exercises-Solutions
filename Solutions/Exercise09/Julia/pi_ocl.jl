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
#          Ported to Julia by Jake Bolewski, Nov 2013

import OpenCL
const cl = OpenCL

# get the directory of this file
# (used for test runner)
src_dir = dirname(Base.source_path())

#
# Some constant values
const INSTEPS = 512*512*512
const ITERS = 262144

# Set some default values:
# Default number of steps (updated later to device prefereable)
const in_nsteps = INSTEPS

# Default number of iterations
const niters = ITERS

# create context, queue and build program
device, ctx, queue = cl.create_compute_context()

kernelsource = open(readall, joinpath(src_dir, "../pi_ocl.cl"))
program = cl.Program(ctx, source=kernelsource) |> cl.build!

# pi is a julia keyword
pi_kernel = cl.Kernel(program, "pi")

# get the max work group size for the kernel pi on the device
work_group_size = device[:max_work_group_size]

# now that we know the size of the work_groups, we can set the number
# of work groups, the actual number of steps, and the step size
nwork_groups = int(in_nsteps / (work_group_size * niters))

if nwork_groups < 1
    # you can get opencl object info through the obj[:symbol] syntax
    # or cl.info(obj, :symbol)
    nwork_groups = device[:max_compute_units]
    work_group_size = in_nsteps / (nwork_groups * niters)
end

nsteps = work_group_size * niters * nwork_groups
step_size = 1.0 / nsteps

# vector to hold partial sum
h_psum = Array(Float32, nwork_groups)

println("$nwork_groups work groups of size $work_group_size.")
println("$nsteps integration steps")

d_partial_sums = cl.Buffer(Float32, ctx, :w, length(h_psum))

# start timer 
rtime = time() 

# Execute the kernel over the entire range of our 1d input data et
# using the maximum number of work group items for this device
# Set the global and local size as tuples
global_size = (nwork_groups * work_group_size,)
local_size  = (work_group_size,)
localmem    = cl.LocalMem(Float32, work_group_size)

cl.call(queue, pi_kernel, global_size, local_size,
        int32(niters), float32(step_size), localmem, d_partial_sums)

cl.copy!(queue, h_psum, d_partial_sums)

# complete the sum and compute final integral value
pi_res = sum(h_psum) * step_size

# stop the timer
rtime = time() - rtime

println("The calculation ran in $rtime secs")
println("pi=$pi_res for $nsteps steps")
