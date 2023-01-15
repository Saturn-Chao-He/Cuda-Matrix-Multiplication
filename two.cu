//******************************************************************************
// Fall 2020
// Assignment: two.cu
// Instructor: Dr. Jiang
// Programmer: Chao He
//******************************************************************************
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>

// Function prototype
void initialData(int *, int *, int *, int);
void MatrixMalOnGPU(const int *, const int *, int *, const int, const int, const int);
__global__ void MatrixMalOnGPU_0(const int *, const int *, int *, const int, const int, const int);
__global__ void MatrixMalOnGPU_1(const int *, const int *, int *, const int, const int, const int);

// Main function
int main(int argc, char **argv)
{
	printf("\n*****************************************************************************************\n");
    printf("\n%s Starting...\n", argv[0]);

	// result[M][S] = a[M][N] * b[N][S]
    int M = 1024, N = 1024, S = 1024;

	// Allocate the space of matrices A and B
	int *a = (int *)malloc(M * N * sizeof(int));
	if (NULL == a)
	{
		printf("the malloc of Matrix a is failed!\n");
		return 0;
    }
	int *b = (int *)malloc(N * S * sizeof(int));
	if (NULL == b)
	{
		printf("the malloc of Matrix b is failed!\n");
		return 0;
	}
	
	// Allocate the space of matrices 
	int *c = (int *)malloc(M * S * sizeof(int));
	if (NULL == c)
	{
		printf("the malloc of Matrix c is failed!\n");
		return 0;
    }

	// Page-locked memroy A B on host
	cudaMallocHost ( (void**)&a, 1024*1024*sizeof(int) );
	cudaMallocHost ( (void**)&b, 1024*1024*sizeof(int) );
	cudaMallocHost ( (void**)&c, 1024*1024*sizeof(int) );

	// Initialize the matrix and print them 
	initialData(a, b, c, 1024*1024);

	//GPU excution
	MatrixMalOnGPU(a, b, c, M, N, S);
	
	// Print the result of C
	printf("\n");
	printf("\nThe result of matrix C:\n");
    for(int i = 0; i < 10; i++)  printf("%d ", c[i]);
    printf("\n");
	for(int i = 1024*1024-10; i < 1024*1024; i++)  printf("%d ", c[i]);  
	printf("\n");


/*
	printf("\nThe result of GPU:\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			printf("%d ", c[i * M + j]);
		}
		printf("-----------[%d]", i);
		printf("\n");
    }
    printf("\n");
*/


    // free host memory
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	printf("\n*****************************************************************************************\n");
    return 0;
}

// Initialize the matrix and print them 
void initialData(int *a, int *b, int *c, int size )
{
    // Fill 1 to matrix A and fill 2 to matrix B
    for (int i = 0; i < 1024*1024; i++)
    {
        a[i] = 1;
        b[i] = 2;
		c[i] = 0;
    }
    // Print A
    printf("\nThe matrix A: \n");
    for(int i = 0; i < 10; i++)  printf("%d ", a[i]);
    printf("\n");
	for(int i = 1024*1024-10; i < 1024*1024; i++)  printf("%d ", a[i]);  
    printf("\n");

    printf("\n");
	// Print B
    printf("\nThe matrix B: \n");
    for(int i = 0; i < 10; i++)  printf("%d ", b[i]);
    printf("\n");
	for(int i = 1024*1024-10; i < 1024*1024; i++)  printf("%d ", b[i]);  
    printf("\n");

	// Print C
	printf("\n");
    printf("\nThe matrix C: \n");
    for(int i = 0; i < 10; i++)  printf("%d ", c[i]);
    printf("\n");
	for(int i = 1024*1024-10; i < 1024*1024; i++)  printf("%d ", c[i]);  
    printf("\n");

    printf("\n");
}

// GPU excution
void MatrixMalOnGPU(const int *host_a, const int *host_b, int *result, const int M, const int N, const int S)
{
	
	// Set up device 0
    int dev_0 = 0;
    cudaDeviceProp deviceProp_0;
    CHECK(cudaGetDeviceProperties(&deviceProp_0, dev_0));
    printf("\nUsing Device %d: %s\n", dev_0, deviceProp_0.name);
	
	// Set up device 1
    int dev_1 = 1;
    cudaDeviceProp deviceProp_1;
    CHECK(cudaGetDeviceProperties(&deviceProp_1, dev_1));
    printf("\nUsing Device %d: %s\n", dev_1, deviceProp_1.name);
	
	// GPU total execution time
	cudaEvent_t gpu_total_Start, gpu_total_Finish;
	float gpu_total_elapsedTime;
	cudaEventCreate(&gpu_total_Start);
	cudaEventCreate(&gpu_total_Finish);
	cudaEventRecord(gpu_total_Start, 0);

	// Maximum number of threads per block:           1024
	// Maximum sizes of each dimension of a block:    1024 x 1024 x 64
	// Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
	// Maximum number of threads per multiprocessor:  2048

	// half of the matrix size
	int nx = 512; // S
	int ny = 1024; // M
	// use 2D grid and 2D blocks
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid( nx/block.x, ny/block.y );


	// device_1
	cudaSetDevice(1);	

	// GPU_1 execution time
	cudaEvent_t gpu_1_Start, gpu_1_Finish;
	float gpu_1_elapsedTime;
	cudaEventCreate(&gpu_1_Start);
	cudaEventCreate(&gpu_1_Finish);
	cudaEventRecord(gpu_1_Start, 0);

	printf("\n");
	printf("Do matrix multiplication on GPU_1...");
    printf("\n");

	MatrixMalOnGPU_1 <<< grid, block >>> (host_a, host_b, result, M, N, S);
	cudaThreadSynchronize();

	cudaEventRecord(gpu_1_Finish, 0);
	cudaEventSynchronize(gpu_1_Finish);
	cudaEventElapsedTime(&gpu_1_elapsedTime, gpu_1_Start, gpu_1_Finish);
	printf("\nThe GPU_1 excution time is %lf ms.\n", gpu_1_elapsedTime);
	//cudaEventDestory(gpu_1_Start);
	//cudaEventDestory(gpu_1_Finish);


	// device_0
	cudaSetDevice(0);	

	// GPU_0 execution time
	cudaEvent_t gpu_0_Start, gpu_0_Finish;
	float gpu_0_elapsedTime;
	cudaEventCreate(&gpu_0_Start);
	cudaEventCreate(&gpu_0_Finish);
	cudaEventRecord(gpu_0_Start, 0);
	
	printf("\n");
	printf("Do matrix multiplication on GPU_0...");
    printf("\n");

	MatrixMalOnGPU_0 <<< grid, block >>> (host_a, host_b, result, M, N, S);
	cudaThreadSynchronize();

	cudaEventRecord(gpu_0_Finish, 0);
	cudaEventSynchronize(gpu_0_Finish);
	cudaEventElapsedTime(&gpu_0_elapsedTime, gpu_0_Start, gpu_0_Finish);
	printf("\nThe GPU_0 excution time is %lf ms.\n", gpu_0_elapsedTime);
	//cudaEventDestory(gpu_0_Start);
	//cudaEventDestory(gpu_0_Finish);

	
	// GPU total execution time
	cudaEventRecord(gpu_total_Finish, 0);
	cudaEventSynchronize(gpu_total_Finish);
	cudaEventElapsedTime(&gpu_total_elapsedTime, gpu_total_Start, gpu_total_Finish);
	printf("\nThe GPU total excution time is %lf ms.\n", gpu_total_elapsedTime);
	//cudaEventDestory(gpu_total_Start);
	//cudaEventDestory(gpu_total_Finish);

	//synchronizaton
	//cudaDeviceSynchronize();

}

// Multiply the matrices on GPU_0
__global__ void MatrixMalOnGPU_0(const int *a_0, const int *b_0, int *result_0, const int M, const int N, const int S)
{
	// global index: threadId = iy * nx + ix = iy * gridDim.x * blockDim.x + ix
	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < M * S)
	{
		// linear global index, so /S to find the row and use %S to find the column
		int row = threadId / S; 
		int column = threadId % S;

		for (int i = 0; i < N; i++)
		{
			result_0[threadId] += a_0[row * N + i] * b_0[i * S + column];
		}
	}
}


// Multiply the matrices on GPU_1
__global__ void MatrixMalOnGPU_1(const int *a_1, const int *b_1, int *result_1, const int M, const int N, const int S)
{
	// global index: threadId = iy * nx + ix = iy * gridDim.x * blockDim.x + ix
	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < M * S) // threadId (0 ~ 512*1024-1)
	{
		// linear global index, so /S to find the row and use %S to find the column
		int row = threadId / S; // (0 ~ 511)
		int column = threadId % S; // (0 ~ 511)

		for (int i = 0; i < N; i++) // i helps iteration to go horizontally in A[] and go vertically in B[]
		{
			result_1[threadId + (512*1024)] += a_1[ ((row + 512)*N ) + i -1 ] * b_1[ column + i * S];
		}
	}
}
