
__kernel void mmul(
    const int Mdim,
    const int Ndim,
    const int Pdim,
    __global float* A,
    __global float* B,
    __global float* C,
    __local float* Bwrk)
{
    int k, j;
    int i    = get_global_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);
    float Awrk[1024];
    float tmp;
    if (i < Ndim) {
        for (k = 0; k < Pdim; k++)
            Awrk[k] = A[i*Ndim+k];

        for (j = 0; j < Mdim; j++) {
            for (k = iloc; k < Pdim; k += nloc)
                Bwrk[k] = B[k*Pdim+j];
            barrier(CLK_LOCAL_MEM_FENCE);
            tmp = 0.0f;
            for (k = 0; k < Pdim; k++)
                tmp += Awrk[k] * Bwrk[k];
            C[i*Ndim+j] = tmp;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
