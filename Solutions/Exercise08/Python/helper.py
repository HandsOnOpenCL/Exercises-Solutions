
from definitions import *
import numpy

#  Function to compute the matrix product (sequential algorithm, dot prod)
def seq_mat_mul_sdot_old (N, A, B, C):
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i*N+k] * B[k*N+j]
            C[i*N+j] = tmp

def seq_mat_mul_sdot (N, A, B, C):
    a2 = A.reshape([N,N]);
    b2 = B.reshape([N,N]);
    c2 = numpy.matmul(a2,b2);
    C  = c2.reshape(N*N);

#  Function to compute errors of the product matrix
def error(N, C1, C2):
    d1 = C1 - C2;
    d2 = (d1 * d1).sum()
    return d2;


# Function to analyze and output results
def results(N, C1, C2, run_time):
    mflops = 2.0 * N * N * N/(1000000.0* run_time)
    print run_time, "seconds at", mflops, "MFLOPS"
    errsq = error(N, C1, C2)
    if (errsq > TOL):
        print "Errors in multiplication:", errsq

## Return 1D array of length N*N initialized to non-constant small values.
## Second variable 'm' is used to allow different initializations for different m's.
## Current implementation simply initializes matrices to sequential values modulo 'm',
##   but one can come up with a more randomized scheme, like using 'm' for seeding
##   random number generator.

def initMat (N, m):
    return numpy.arange(N*N, dtype=numpy.float32) % m;
