// CUDA implementation of 1d_stencil
// compile: $ nvcc -o t280 t280.cu
//          $ cuda-memcheck ./t280
//
#define N 30
#define THREADS_PER_BLOCK 10
#define BLOCK_SIZE THREADS_PER_BLOCK
#define RADIUS 3

#include <stdio.h>

void random_ints(int *var, int n) // Attribue une valeur à toutes le composantes des variables
{
    int i;
    for (i = 0; i < n; i++)
        var[i] = 1;
}

// 1d_stencil function runnned on multi_block multi_thread
__global__ void stencil_1d(int *in, int *out) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];  // shared memory in block 

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = (gindex >= RADIUS)?in[gindex - RADIUS]:0; 
        temp[lindex + BLOCK_SIZE] = ((gindex + BLOCK_SIZE)<N)?in[gindex + BLOCK_SIZE]:0; 

    __syncthreads(); 
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];

    out[gindex] = result;
}

int main()
{
    int size = N * sizeof(int);

    int *in, *out; 
    int *d_in, *d_out;  

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