//Cuda hello world
#include<stdio.h>
#define N 10
#define THREADS_PER_BLOCK 1
#define BLOCK_SIZE THREADS_PER_BLOCK

// calculation of loss
__global__ void cal_loss(float *err, float *label, int n){

	printf("threadIdx:(%d) blockIdx:(%d)\n "
	, threadIdx.x, blockIdx.x);
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	//for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) { 
	//	err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]); // calculation of error
	//}
}

void init_float(float * arr, int size){
	for (int i = 0; i < size; ++i)
	{
		if(i == 3)
			arr[i] = 1;
		else
			arr[i] = 0;
	}
}


int main()
{
	// host data
	int size = 10 * sizeof(float);
	float *label = (float*)malloc(size);
	init_float(label, 10);
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

    free(label); free(err);
  
    cudaFree(d_label); cudaFree(d_err);

    return 0;
}

