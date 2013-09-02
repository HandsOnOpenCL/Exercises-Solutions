//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library for the multiplication driver
//
//  PURPOSE: This is a simple set of functions to manipulate
//           matrices used with the multiplcation driver.
//
//  USAGE:   The matrices are square and the order is
//           set as a defined constant, ORDER.
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------

#include "matmul.hpp"

//------------------------------------------------------------------------------
//
//  Function to compute the matrix product (sequential algorithm, dot prod)
//
//------------------------------------------------------------------------------

void seq_mat_mul_sdot(int Mdim, int Ndim, int Pdim, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    int i, j, k;
    float tmp;

    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++) {
            tmp = 0.0f;
            for (k = 0; k < Pdim; k++) {
                /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
                tmp += A[i*Ndim+k] * B[k*Pdim+j];
            }
            C[i*Ndim+j] = tmp;
        }
    }
}

//------------------------------------------------------------------------------
//
//  Function to initialize the input matrices A and B
//
//------------------------------------------------------------------------------
void initmat(int Mdim, int Ndim, int Pdim, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    int i, j;

    /* Initialize matrices */

	for (i = 0; i < Ndim; i++)
		for (j = 0; j < Pdim; j++)
			A[i*Ndim+j] = AVAL;

	for (i = 0; i < Pdim; i++)
		for (j = 0; j < Mdim; j++)
			B[i*Pdim+j] = BVAL;

	for (i = 0; i < Ndim; i++)
		for (j = 0; j < Mdim; j++)
			C[i*Ndim+j] = 0.0f;
}

//------------------------------------------------------------------------------
//
//  Function to set a matrix to zero
//
//------------------------------------------------------------------------------
void zero_mat (int Ndim, int Mdim, std::vector<float>& C)
{
    int i, j;

	for (i = 0; i < Ndim; i++)
		for (j = 0; j < Mdim; j++)
			C[i*Ndim+j] = 0.0f;
}

//------------------------------------------------------------------------------
//
//  Function to fill Btrans(Mdim,Pdim)  with transpose of B(Pdim,Mdim)
//
//------------------------------------------------------------------------------
void trans(int Pdim, int Mdim, std::vector<float>& B, std::vector<float>& Btrans)
{
    int i, j;

	for (i = 0; i < Pdim; i++)
		for (j = 0; j < Mdim; j++)
		    Btrans[j*Pdim+i] = B[i*Mdim+j];
}

//------------------------------------------------------------------------------
//
//  Function to compute errors of the product matrix
//
//------------------------------------------------------------------------------
float error(int Mdim, int Ndim, int Pdim, std::vector<float>& C)
{
   int i,j;
   float cval, errsq, err;
   cval = (float) Pdim * AVAL * BVAL;
   errsq = 0.0f;

    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++) {
            err = C[i*Ndim+j] - cval;
            errsq += err * err;
        }
    }
    return errsq;
}

//------------------------------------------------------------------------------
//
//  Function to analyze and output results
//
//------------------------------------------------------------------------------
void results(int Mdim, int Ndim, int Pdim, std::vector<float>& C, double run_time)
{

    float mflops;
    float errsq;
    mflops = 2.0 * Mdim * Ndim * Pdim/(1000000.0f * run_time);
    printf(" %.2f seconds at %.1f MFLOPS \n",  run_time,mflops);
    errsq = error(Mdim, Ndim, Pdim, C);
    if (isnan(errsq) || errsq > TOL)
           printf("\n Errors in multiplication: %f\n",errsq);
}

