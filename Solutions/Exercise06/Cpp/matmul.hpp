//------------------------------------------------------------------------------
//
//  Include fle for the Matrix Multiply test harness
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------

#ifndef __MULT_HDR
#define __MULT_HDR

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"


#include "matrix_lib.hpp"

//------------------------------------------------------------------------------
//  functions from ../Common
//------------------------------------------------------------------------------
extern double wtime();   // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#define ORDER    1024    // Order of the square matrices A, B, and C
#define AVAL     3.0     // A elements are constant and equal to AVAL
#define BVAL     5.0     // B elements are constant and equal to BVAL
#define TOL      (0.001) // tolerance used in floating point comparisons
#define DIM      2       // Max dim for NDRange
#define COUNT    1       // number of times to do each multiplication
#define SUCCESS  1
#define FAILURE  0

#endif
