//------------------------------------------------------------------------------
//
// Name:       vadd_chain.cpp
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
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
    std::vector<float> h_c (LENGTH, 0xdeadbeef);   // c vector (result)
    std::vector<float> h_d (LENGTH, 0xdeadbeef);   // d vector (result)
    std::vector<float> h_e (LENGTH);               // e vector
    std::vector<float> h_f (LENGTH, 0xdeadbeef);   // f vector (result)
    std::vector<float> h_g (LENGTH);               // g vector

    cl::Buffer d_a;                       // device memory used for the input  a vector
    cl::Buffer d_b;                       // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the output c vector
    cl::Buffer d_d;                       // device memory used for the output d vector
    cl::Buffer d_e;                       // device memory used for the input e vector
    cl::Buffer d_f;                       // device memory used for the output f vector
    cl::Buffer d_g;                       // device memory used for the input g vector

    // Fill vectors a and b with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i]  = rand() / (float)RAND_MAX;
        h_b[i]  = rand() / (float)RAND_MAX;
        h_e[i]  = rand() / (float)RAND_MAX;
        h_g[i]  = rand() / (float)RAND_MAX;
    }

    try 
    {
    	// Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("vadd_chain.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
 
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

        d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true);
        d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_e   = cl::Buffer(context, h_e.begin(), h_e.end(), true);
        d_g   = cl::Buffer(context, h_g.begin(), h_g.end(), true);

        d_c  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_d  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_f  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)), 
            d_a,
            d_b,
            d_c,
            count);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)), 
            d_e,
            d_c,
            d_d,
            count);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)), 
            d_g,
            d_d,
            d_f,
            count);

        cl::copy(queue, d_f, h_f.begin(), h_f.end());

        // Test the results
        int correct = 0;
        float tmp;
        for(int i = 0; i < count; i++)
        {
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i];     // assign element i of a+b+e+g to tmp
            tmp -= h_f[i];                               // compute deviation of expected and output result
            if(tmp*tmp < TOL*TOL)                        // correct if square deviation is less than tolerance squared
                correct++;
            else {
                printf(" tmp %f h_a %f h_b %f h_e %f h_g %f h_f %f\n",tmp, h_a[i], h_b[i], h_e[i], h_g[i], h_f[i]);
            }
        }
        
        // summarize results
        printf("C = A+B+E+G:  %d out of %d results were correct.\n", correct, count);
        
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
