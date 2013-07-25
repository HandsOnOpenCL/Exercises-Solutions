
from definitions import *
import numpy

#  Function to compute the matrix product (sequential algorithm, dot prod)
def seq_mat_mul_sdot(Mdim, Ndim, Pdim, A, B, C):
    for i in range(Ndim):
        for j in range(Mdim):
            tmp = 0.0
            for k in range(Pdim):
                tmp += A[i*Ndim+k] * B[k*Pdim+j]
            C[i*Ndim+j] = tmp

#  Function to compute errors of the product matrix
def error(Mdim, Ndim, Pdim, C):
   cval = float(Pdim) * AVAL * BVAL
   errsq = 0.0
   for i in range(Ndim):
       for j in range(Mdim):
            err = C[i*Ndim+j] - cval
            errsq += err * err
   return errsq;


# Function to analyze and output results
def results(Mdim, Ndim, Pdim, C, run_time):
    mflops = 2.0 * Mdim * Ndim * Pdim/(1000000.0* run_time)
    print run_time, "seconds at", mflops, "MFLOPS"
    errsq = error(Mdim, Ndim, Pdim, C)
    if numpy.isnan(errsq) or errsq > TOL:
        print "Errors in multiplication:", errsq
