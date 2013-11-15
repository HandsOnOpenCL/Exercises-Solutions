#
# Device Info
#
# Function to output key parameters about the input OpenCL device
#
# History: C version written by Tim Mattson, June 2010
#          Ported to Python by Tom Deakin, July 2013
#          Ported to Julia  by Jake Bolewski, Nov 2013

import OpenCL

function output_device_info(d::OpenCL.Device)
    n  = d[:name]
    dt = d[:device_type]
    v  = d[:platform][:vendor] 
    mc = d[:max_compute_units] 
    str = "Device is $n $dt from $v with a max of $mc compute units" 
    println(str)
end

