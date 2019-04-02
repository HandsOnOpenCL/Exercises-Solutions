#
# Display Device Information
#
# Script to print out some information about the OpenCL devices
# and platforms available on your system
#
# History: C++ version written by Tom Deakin, 2012
#          Ported to Python by Tom Deakin, July 2013
#

import OpenCL
const cl = OpenCL

# create a list of all the platform ids
platforms = cl.platforms()

println("\nNumber of OpenCL platforms: $(length(platforms))")
println("\n-----------------------------\n")

# info for each platform
for p in platforms
    
    # print out some info
    @printf("Platform: %s\n", p[:name])
    @printf("Vendor:   %s\n", p[:vendor])
    @printf("Version:  %s\n", p[:version])

    # discover all the devices
    devices = cl.devices(p)
    @printf("Number of devices: %s\n", length(devices))

    for d in devices
        println("\t-----------------------------")
        # Print out some information about the devices
        @printf("\t\tName: %s\n", d[:name])
        @printf("\t\tVersion: %s\n", d[:version]) 
        @printf("\t\tMax. Compute Units: %s\n", d[:max_compute_units])
        @printf("\t\tLocal Memory Size: %i KB\n", d[:local_mem_size] / 1024)
        @printf("\t\tGlobal Memory Size: %i MB\n", d[:global_mem_size] / (1024^2))
        @printf("\t\tMax Alloc Size: %i MB\n", d[:max_mem_alloc_size] / (1024^2))
        @printf("\t\tMax Work-group Size: %s\n", d[:max_work_group_size])

        # Find the maximum dimensions of the work-groups
        dim = d[:max_work_item_size]
        @printf("\t\tMax Work-item Dims: %s\n", dim)
        println("\t-----------------------------")
    end

    print("\n-------------------------")
end

