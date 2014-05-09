
__kernel void mmul(
    const int Mdim,
    const int Ndim,
    const int Pdim,
    __global float* A,
    __global float* B,
    __global float* C)
{
    int k, j;
    int i = get_global_id(0);
    float Awrk[1024];
    float tmp;
    if (i < Ndim) {
        for (k = 0; k < Pdim; k++)
            Awrk[k] = A[i*Ndim+k];

        for (j = 0; j < Mdim; j++) {
            tmp = 0.0f;
            for (k = 0; k < Pdim; k++)
                tmp += Awrk[k] * B[k*Pdim+j];
            C[i*Ndim+j] = tmp;
        }
    }
}
