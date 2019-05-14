//Cuda hello world
#include<stdio.h>
#define N 10
#define THREADS_PER_BLOCK 1
#define BLOCK_SIZE THREADS_PER_BLOCK

// calculation of loss
__global__ void cal_loss(int *err, int *label, int N) {

	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
	"gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
	blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
	gridDim.x,gridDim.y,gridDim.z);
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	//for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) { 
	//	err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]); // calculation of error
	//}
	return 0;
}

int main()
{
	// host data
	int size = 10 * sizeof(float);
	//float *label = [0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f];
	float *label;
	float *err;
    err = (float*)malloc(size);
    label = (float*)malloc(size);

    // copy data to device
    float *d_label, *d_err;
    cudaMalloc(&d_label, size);
    cudaMalloc(&d_err, sizeof(float) * 10);
    cudaMemcpy(d_label, label, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_err, err, size, cudaMemcpyHostToDevice);
 
    cal_loss <<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_err, d_label, 10);// kernel function
  
    //cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost); // copy the result from GPU to CPU

    //for(int i=0; i<N; i++)
    //    printf("%i ---i=%i \n", out[i], i);

    free(label); free(err);
  
    cudaFree(d_label); cudaFree(d_err);

    return 0;
}

