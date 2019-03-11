
from definitions import *
import numpy

#  Function to compute the matrix product (sequential algorithm, dot prod)
def seq_mat_mul_sdot( Ndim, A, B, C):
    for i in range(Ndim):
        for j in range(Ndim):
            tmp = 0.0
            for k in range(Ndim):
                tmp += A[i*Ndim+k] * B[k*Ndim+j]
            C[i*Ndim+j] = tmp

#  Function to compute errors of the product matrix
def error( Ndim, C):
   cval = float(Ndim) * AVAL * BVAL
   errsq = 0.0
   for i in range(Ndim):
       for j in range(Ndim):
            err = C[i*Ndim+j] - cval
            errsq += err * err
   return errsq;


# Function to analyze and output results
def results( Ndim, C, run_time):
    mflops = ( 2.0 * (Ndim**(3)) )/(1000000.0* run_time)
    print run_time, "seconds at", mflops, "MFLOPS"
    errsq = error( Ndim, C)
    if numpy.isnan(errsq) or errsq > TOL:
        print "Errors in multiplication:", errsq
