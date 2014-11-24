//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library include file (function prototypes)
//
//  HISTORY: Written by Tim Mattson, August 2010 
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Ported by Tom Deakin, July 2013
//
//------------------------------------------------------------------------------

#ifndef __MATRIX_LIB_HDR
#define __MATRIX_LIB_HDR


//------------------------------------------------------------------------------
//
//  Function to compute the matrix product (sequential algorithm, dot producdt)
//
//------------------------------------------------------------------------------
void seq_mat_mul_sdot(int N, float *A, float *B, float *C);

//------------------------------------------------------------------------------
//
//  Function to initialize the input matrices A and B
//
//------------------------------------------------------------------------------
void initmat(int N, float *A, float *B, float *C);

//------------------------------------------------------------------------------
//
//  Function to set a matrix to zero 
//
//------------------------------------------------------------------------------
void zero_mat (int N, float *C);

//------------------------------------------------------------------------------
//
//  Function to fill Btrans(Mdim,Pdim)  with transpose of B(Pdim,Mdim)
//
//------------------------------------------------------------------------------
void trans(int N, float *B, float *Btrans);

//------------------------------------------------------------------------------
//
//  Function to compute errors of the product matrix
//
//------------------------------------------------------------------------------
float error(int N, float *C);


//------------------------------------------------------------------------------
//
//  Function to analyze and output results 
//
//------------------------------------------------------------------------------
void results(int N, float *C, double run_time);
    
#endif
