
from definitions import *

#  Function to compute the matrix product (sequential algorithm, dot prod)
def seq_mat_mul_sdot(N, A, B, C):
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i*N+k] * B[k*N+j]
            C[i*N+j] = tmp

#  Function to compute errors of the product matrix
def error(N, C):
   cval = float(N) * AVAL * BVAL
   errsq = 0.0
   for i in range(N):
       for j in range(N):
            err = C[i*N+j] - cval
            errsq += err * err
   return errsq;


# Function to analyze and output results
def results(N, C, run_time):
    mflops = 2.0 * N * N * N/(1000000.0* run_time)
    print run_time, "seconds at", mflops, "MFLOPS"
    errsq = error(N, C)
    if (errsq > TOL):
        print "Errors in multiplication:", errsq
