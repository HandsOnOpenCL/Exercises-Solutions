//------------------------------------------------------------------------------
//
// Name:       vadd_three.cpp
//
// Purpose:    Elementwise addition of three vectors (d = a + b + c)
//
//                   d = a + b + c
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include "err_code.h"

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

int main(void)
{
    std::vector<float> h_a(LENGTH);                // a vector
    std::vector<float> h_b(LENGTH);                // b vector
    std::vector<float> h_c(LENGTH);                // c vector
    std::vector<float> h_d (LENGTH, 0xdeadbeef);   // d vector (result)

    cl::Buffer d_a;                       // device memory used for the input  a vector
    cl::Buffer d_b;                       // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the input c vector
    cl::Buffer d_d;                       // device memory used for the output d vector

    // Fill vectors a and b with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i]  = rand() / (float)RAND_MAX;
        h_b[i]  = rand() / (float)RAND_MAX;
        h_c[i]  = rand() / (float)RAND_MAX;
    }

    try
    {
    	// Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("vadd_abc.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor

        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

        d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true);
        d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_c   = cl::Buffer(context, h_c.begin(), h_c.end(), true);

        d_d  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)),
            d_a,
            d_b,
            d_c,
            d_d,
            count);

        cl::copy(queue, d_d, h_d.begin(), h_d.end());

        // Test the results
        int correct = 0;
        float tmp;
        for(int i = 0; i < count; i++)
        {
            tmp = h_a[i] + h_b[i] + h_c[i];              // assign element i of a+b+c to tmp
            tmp -= h_d[i];                               // compute deviation of expected and output result
            if(tmp*tmp < TOL*TOL)                        // correct if square deviation is less than tolerance squared
                correct++;
            else {
                printf(" tmp %f h_a %f h_b %f h_c %f h_d %f\n",tmp, h_a[i], h_b[i], h_c[i], h_d[i]);
            }
        }

        // summarize results
        printf("D = A+B+C:  %d out of %d results were correct.\n", correct, count);

    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
           << ")"
           << std::endl;
    }
}
