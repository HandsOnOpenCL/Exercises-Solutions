# Pi reduction - vectorized
#
# Numeric integration to estimate pi
# Asks the user to select a device at runtime
# Vector size must be present as a CLI argument
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

# Some constant values
INSTEPS = 512 * 512 * 512
ITERS = -1
WGS = -1
NAME = ""

if length(ARGS) < 1 
    info("Usage: julia pi_vocl.jl [num] (where num = 1, 4, or 8)")
    exit(1)
end
vector_size = int(ARGS[1])

if vector_size == 1
        ITERS = 262144
        WGS = 8
elseif vector_size == 4 
        ITERS = 65536 # (262144/4)
        WGS = 32
elseif vector_size == 8
        ITERS = 32768 # (262144/8)
        WGS = 64
else
    warn("Invalid vector size")
    exit(1)
end

# Set some default values:
# Default number of steps (updated later to device prefereable)
in_nsteps = INSTEPS

# Default number of iterations
niters = ITERS
work_group_size = WGS

# Create context, queue and build program

#---------------------------------------------
# Uncomment to switch between gpu/cpu devices
#---------------------------------------------
#device = first(cl.devices(:gpu))
#device = first(cl.devices(:cpu))
device, ctx, queue = cl.create_some_context()


kernelsource = open(readall, joinpath(src_dir, "../pi_vocl.cl"))
program = cl.Program(ctx, source=kernelsource) |> cl.build!

if vector_size == 1
    pi_kernel = cl.Kernel(program, "pi")
elseif vector_size == 4
    pi_kernel = cl.Kernel(program, "pi_vec4")
elseif vector_size == 8
    pi_kernel = cl.Kernel(program, "pi_vec8")
end

# Now that we know the size of the work_groups, we can set the number of work
# groups, the actual number of steps, and the step size
nwork_groups = int(in_nsteps / (work_group_size * niters))

# get the max work group size for the kernel on our device
if vector_size == 1
    max_size = cl.work_group_info(pi_kernel, :size, device)
elseif vector_size == 4
    max_size = cl.work_group_info(pi_kernel, :size, device)
elseif vector_size == 8
    max_size = cl.work_group_info(pi_kernel, :size, device)
end

if max_size > work_group_size
    work_group_size = max_size
    nwork_groups = int(in_nsteps / (work_group_size * niters))
end

if nwork_groups < 1
    nwork_groups = device[:max_compute_units]
    work_group_size = int(in_nsteps / (nwork_groups * niters))
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
