//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum d = a+b+c
//
// input: a, b and c float vectors of length count
//
// output: d float vector of length count holding the sum a + b + c
//

__kernel void vadd(
   __global float* a,
   __global float* b,
   __global float* c,
   __global float* d,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)  {
       d[i] = a[i] + b[i] + c[i];
   }
}

