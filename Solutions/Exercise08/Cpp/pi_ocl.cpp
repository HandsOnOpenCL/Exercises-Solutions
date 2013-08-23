//------------------------------------------------------------------------------
//
// Name:       pi_ocl.cpp
// 
// Purpose:    Numeric integration to estimate pi
//
// HISTORY:    Written by Tim Mattson, May 2010 
//        :    Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "util.hpp"


#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


#define INSTEPS (512*512*512)
#define ITERS (262144)

int main(void)
{
    float *h_psum;					// vector to hold partial sum
    int in_nsteps = INSTEPS;		// default number of steps (updated later to device prefereable)
    int niters = ITERS;				// number of iterations
    int nsteps;
    float step_size;
    ::size_t nwork_groups;
    ::size_t max_size, work_group_size = 8;
    float pi_res;

	cl::Buffer d_partial_sums;

	try
	{
        // Create a context
        cl::Context context(DEVICE);

        // Create the program object
		cl::Program program(context, util::loadProgram("../pi_ocl.cl"), true);

		// Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel object for quering information
		cl::Kernel ko_pi(program, "pi");
        // Get the device we are using
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];

        // Get the work group size
		work_group_size = ko_pi.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        //printf("wgroup_size = %lu\n", work_group_size);

		auto pi = cl::make_kernel<int, float, cl::LocalSpaceArg, cl::Buffer>(program, "pi");

		// Now that we know the size of the work_groups, we can set the number of work
		// groups, the actual number of steps, and the step size
		nwork_groups = in_nsteps/(work_group_size*niters);

		if ( nwork_groups < 1) {
			nwork_groups = 
				device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			work_group_size=in_nsteps / (nwork_groups*niters);
		}

		nsteps = work_group_size * niters * nwork_groups;
		step_size = 1.0f/static_cast<float>(nsteps);
		std::vector<float> h_psum(nwork_groups);

		printf(
			" %d work groups of size %d.  %d Integration steps\n", 
			(int)nwork_groups, 
			(int)work_group_size,
			nsteps);

		d_partial_sums = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups);

		util::Timer timer;

		// Execute the kernel over the entire range of our 1d input data set
		// using the maximum number of work group items for this device
		pi(
			cl::EnqueueArgs(
				queue,
				cl::NDRange(nwork_groups * work_group_size),
				cl::NDRange(work_group_size)), 
			niters,
			step_size,
			cl::Local(sizeof(float) * work_group_size),
			d_partial_sums);

		cl::copy(queue, d_partial_sums, begin(h_psum), end(h_psum));

		// complete the sum and compute final integral value
		pi_res = 0.0f;
		for (unsigned int i = 0; i< nwork_groups; i++) {
			pi_res += h_psum[i];
		}
		pi_res = pi_res * step_size;
	
		//rtime = wtime() - rtime;
		double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.;
		printf("\nThe calculation ran in %lf seconds\n", rtime);
		printf(" pi = %f for %d steps\n", pi_res, nsteps);

	}
	catch (cl::Error err) {
		std::cout << "Exception\n";
		std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
           << ")"
           << std::endl;
	}
}
