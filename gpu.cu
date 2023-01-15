//******************************************************************************
// Fall 2020
// Assignment: gpu.cu
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
cudaError_t MatrixMalOnGPU(const int *, const int *, int *, const int, const int, const int);
__global__ void MatrixMalOnGPUKernel(const int *, const int *, int *, const int, const int, const int);

// Main function
int main(int argc, char **argv)
{
	printf("\n*****************************************************************************************\n");
    printf("\n%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

	// result[M][S] = a[M][N] * b[N][S]
    int M = 1024, N = 1024, S = 1024;

	// Allocate the space of matrices A and B
	int * a = (int *)malloc(M * N * sizeof(int));
	if (NULL == a)
	{
		printf("the malloc of Matrix a is failed!\n");
		return 0;
    }
	int * b = (int *)malloc(N * S * sizeof(int));
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

	// Initialize the matrix and print them 
	initialData(a, b, c, 1024*1024);

	//GPU excution
	cudaError_t cudaStatus = MatrixMalOnGPU(a, b, c, M, N, S);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MatrixMalOnGPU failed!");
		return 0;
    }
	
	// Print the result of C
	printf("\n");
	printf("\nThe result of matrix C:\n");
    for(int i = 0; i < 10; i++)  printf("%d ", c[i]);
    printf("\n");
	for(int i = 1024*1024-10; i < 1024*1024; i++)  printf("%d ", c[i]);  
    printf("\n");

    // free host memory
    free(a);
    free(b);
	free(c);
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
cudaError_t MatrixMalOnGPU(const int *a, const int *b, int *result, const int M, const int N, const int S)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	
	// Check the device
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!\n");		
	}

	// Malloc device global memory time
	cudaEvent_t mallocStart, mallocFinish;
	float mallocElapsedTime;
	cudaEventCreate(&mallocStart);
	cudaEventCreate(&mallocFinish);
	cudaEventRecord(mallocStart, 0);

	// Malloc device global memory 
	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_a failed!\n");		
	}
	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_b failed!\n");		
	}
	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_result failed!\n");		
	}

	cudaEventRecord(mallocFinish, 0);
	cudaEventSynchronize(mallocFinish);
	cudaEventElapsedTime(&mallocElapsedTime, mallocStart, mallocFinish);
	printf("\nThe time of memory allocating is %lf ms.\n", mallocElapsedTime);


	// transfer data time
	cudaEvent_t transferStart, transferFinish;
	float transferElapsedTime;
	cudaEventCreate(&transferStart);
	cudaEventCreate(&transferFinish);
	cudaEventRecord(transferStart, 0);
	// transfer data from host to device
	cudaStatus = cudaMemcpy(dev_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudamemcpy dev_a failed!\n");		
	}
	cudaStatus = cudaMemcpy(dev_b, b, N * S * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dev_b failed!\n");		
	}

	cudaEventRecord(transferFinish, 0);
	cudaEventSynchronize(transferFinish);
	cudaEventElapsedTime(&transferElapsedTime, transferStart, transferFinish);
	printf("\nThe time of data transfering is %lf ms.\n", transferElapsedTime);

	// GPU execution time
	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);

	// Maximum number of threads per block:           1024
	// Maximum sizes of each dimension of a block:    1024 x 1024 x 64
	// Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
	// Maximum number of threads per multiprocessor:  2048
	
	// matrix size
	int nx = 1024; // S
	int ny = 1024; // M
	// use 2D grid and 2D blocks
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid( nx/block.x, ny/block.y );

	printf("\n");
	printf("Do matrix multiplication on GPU...");
    printf("\n");

	MatrixMalOnGPUKernel <<< grid, block >>> (dev_a, dev_b, dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe GPU excution time is %lf ms.\n", elapsedTime);

	// Check Kernel launch
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));
	}

	// Check synchronizaton
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);
	}

    // Copy kernel result back to host side
	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy result failed!\n");
    }
    
    // Free device global memory
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_result));

	return cudaStatus;
}

// Multiply the matrices on GPU
__global__ void MatrixMalOnGPUKernel(const int *a, const int *b, int *result, const int M, const int N, const int S)
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
			result[threadId] += a[row * N + i] * b[i * S + column];
		}
	}
}
