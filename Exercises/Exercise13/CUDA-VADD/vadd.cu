//------------------------------------------------------------------------------
//
// Name:       vadd.cu
// 
// Purpose:    CUDA implementation of VADD
//
// HISTORY:    Written by Tom Deakin and Simon McIntosh-Smith, August 2013
//
//------------------------------------------------------------------------------

#include <stdio.h>
#include <cuda.h>

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

/*************************************************************************************
 * CUDA kernel
 ************************************************************************************/

__global__ void vadd(const float* a,
					 const float* b,
					       float* c,
					 const unsigned int count)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < count) {
		c[i] = a[i] + b[i];
	}
}

/*************************************************************************************
 * Main function
 ************************************************************************************/

int main(void)
{
    float        h_a[LENGTH];       // a vector
    float        h_b[LENGTH];       // b vector
    float        h_c[LENGTH];       // c vector (a+b) returned from the compute device
    float *d_a, *d_b, *d_c;         // CUDA memory
    unsigned int correct;           // number of correct results

    // Fill vectors a and b with random float values
    int i = 0;
    int count = LENGTH;
    for(i = 0; i < count; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate CUDA memory
    cudaMalloc(&d_a, sizeof(float) * LENGTH);
    cudaMalloc(&d_b, sizeof(float) * LENGTH);
    cudaMalloc(&d_c, sizeof(float) * LENGTH);

    // Write buffers a and b to GPU memory
    cudaMemcpy(d_a, h_a, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);

    dim3 numBlocks(LENGTH);
    dim3 numThreads(1);
    vadd<<<numBlocks, numThreads>>>(d_a, d_b, d_c, LENGTH);

    // Copy result array back to host memory
    cudaMemcpy(h_c, d_c, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);

    // Test the results
    correct = 0;
    float tmp;
    
    for(i = 0; i < count; i++)
    {
        tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
        tmp -= h_c[i];             // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
        }
    }
    
    // summarize results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

	return EXIT_SUCCESS;
}
