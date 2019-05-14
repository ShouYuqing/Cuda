//Cuda hello world
#include<stdio.h>
#define N 10
#define THREADS_PER_BLOCK 1
#define BLOCK_SIZE THREADS_PER_BLOCK

// calculation of loss
__global__ void cal_loss(int *error, int *label, int N) {
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) { 
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]); // calculation of error
	}
	return 0;
}

int main()
{
	int size = 10 * sizeof(float);
	float *label = [0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f];
    float *d_label;
    cudaMalloc(&d_label, sizeof(float) * size);
    cudaMemcpy(d_label, label, size, size, cudaMemcpyHostToDevice);




    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    in = (int*)malloc(size); random_ints(in, N);
    out = (int*)malloc(size);
    
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice); // copy memory from CPU to GPU
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);
  
    stencil_1d <<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_in, d_out);// kernel function
  
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost); // copy the result from GPU to CPU

    for(int i=0; i<N; i++)
        printf("%i ---i=%i \n", out[i], i);

    free(in); free(out);
  
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}

