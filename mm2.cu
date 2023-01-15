//******************************************************************************
// Fall 2020
// Assignment: mm2.cu
// Instructor: Dr. Jiang
// Programmer: Chao He
//******************************************************************************
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>

// Function prototype
void initialData(int *, int *, int);
void MatrixMulOnCPU(const int *,const int *, int *,const int,const int,const int);
cudaError_t MatrixMalOnGPU(const int *, const int *, int *, const int, const int, const int);
template<int>
__global__ void MatrixMultWithSharedKernel(const int *, const int *, int *, const int, const int, const int);

// Main function
int main(int argc, char **argv)
{
    printf("\n%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

	// result[M][S] = a[M][N] * b[N][S]
    int M = 16, N = 16, S = 16;

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
	//Store the results of CPU and GPU
	int * cpuResult = (int *)malloc(M * S * sizeof(int));
	if (NULL == cpuResult)
	{
		printf("the malloc of Matrix cpuResult is failed!\n");
		return 0;
    }
    
	int * gpuResult = (int *)malloc(M * S * sizeof(int));
	if (NULL == cpuResult)
	{
		printf("the malloc of Matrix gpuResult is failed!\n");
		return 0;
	}

	// Initialize the matrix and print them 
	initialData(a, b, 256 );

	// The time of CPU excution
	clock_t start, finish;
	double totalTime = 0.0;
	start = clock();

	//CPU excution
	MatrixMulOnCPU(a, b, cpuResult, M, N, S);
	finish = clock();
	totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\nThe CPU excution time is %lf seconds.\n", totalTime);

	//GPU excution
	cudaError_t cudaStatus = MatrixMalOnGPU(a, b, gpuResult, M, N, S);
	
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MatrixMalOnGPU failed!");
		return 0;
    }
    
	// Print the result
	printf("\nThe result of CPU:\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			printf("%d ", cpuResult[i * M + j]);
		}
		printf("\n");
	}
	printf("\nThe result of GPU:\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			printf("%d ", gpuResult[i * M + j]);
		}
		printf("\n");
    }
    printf("\n");

	//Check the results of CPU and GPU
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			if (cpuResult[i * M + j] != gpuResult[i * M + j])
			{
				printf("the Results are not equal!\n");
				return 0;
			}
		}
	}

    // free host memory
    free(a);
    free(b);
    free(cpuResult);
    free(gpuResult);
    return 0;
}

// Initialize the matrix and print them 
void initialData(int *a, int *b, int size )
{
    // Fill 1 to matrix A and fill 2 to matrix B
    for (int i = 0; i < 256; i++)
    {
        a[i] = 1;
        b[i] = 2;
    }
    // Print A and B
    printf("\nThe matrix A: \n");
    for(int i = 0; i < 256; i++)
    {
        printf("%d ", a[i]);
        if( (i+1) % 16 == 0 )   
            printf("\n");
    }
    printf("\n");
    printf("\nThe matrix B: \n");
    for(int i = 0; i < 256; i++)
    {
        printf("%d ", b[i]);
        if( (i+1) % 16 == 0 )   
            printf("\n");
    }

    printf("\n");
    return;
}

// Matrix multiplication on CPU
void MatrixMulOnCPU(const int * a,const int * b, int *result,const int M,const int N,const int S)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			int index = i * S + j;
			result[index] = 0;

			for (int k = 0; k < N; k++)
			{
				result[index] += a[i * N + k] * b[k * S + j];
			}
		}
	}
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

	// Excution time
	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);
	
	// Invoke kernel
    dim3 dimGrid(1, 1);
    dim3 dimBlock(16, 16);
	MatrixMultWithSharedKernel<16> <<< dimGrid, dimBlock >>>(dev_a, dev_b, dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe GPU excution time is %lf seconds.\n", elapsedTime / 1000.0);

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

// Multiply the matrices on GPU using shared memory
template<int TILE_SIZE>
__global__ void MatrixMultWithSharedKernel(const int *a, const int *b, int *result, const int M, const int N, const int S)
{
	if ((threadIdx.y + blockIdx.y * blockDim.y) * S + blockIdx.x * blockDim.x + threadIdx.x >= M * S)
	{
		return;
	}

	const int begin_a = blockIdx.y * blockDim.y * N;
	const int end_a = begin_a + N - 1;
	const int step_a = blockDim.x;

	const int begin_b = blockIdx.x * blockDim.x;
	const int step_b = blockDim.y * S;

	int result_temp = 0;

	int index_a;
	int index_b;
	for (index_a = begin_a, index_b = begin_b;
		 index_a < end_a; 
		 index_a += step_a, index_b += step_b)
	{
		__shared__ int SubMat_A[TILE_SIZE][TILE_SIZE];
		__shared__ int SubMat_B[TILE_SIZE][TILE_SIZE];

		SubMat_A[threadIdx.y][threadIdx.x] = a[index_a + threadIdx.y * N + threadIdx.x];
		SubMat_B[threadIdx.y][threadIdx.x] = b[index_b + threadIdx.y * S + threadIdx.x];

		__syncthreads();

		for (int i = 0; i < TILE_SIZE; i++)
		{
			result_temp += SubMat_A[threadIdx.y][i] * SubMat_B[i][threadIdx.x];
		}

		__syncthreads();
	}

	int begin_result = blockIdx.y * blockDim.y * S + begin_b;
	result[begin_result + threadIdx.y * S + threadIdx.x] = result_temp;
}
