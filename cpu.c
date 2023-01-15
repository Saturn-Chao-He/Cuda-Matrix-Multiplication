//******************************************************************************
// Fall 2020
// Assignment: cpu.c
// Instructor: Dr. Jiang
// Programmer: Chao He
//******************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h> // For Mac using #include <sys/malloc.h>
#include <sys/time.h> 
#include <unistd.h>

// strutures in <sys/malloc.h>
/*
struct timeval {
    time_t tv_sec;	 //seconds 
    long tv_usec;	 //microseconds 
};
 
struct timezone {
    int tz_minuteswest;
    int tz_dsttime;

*/

// Function prototype
void initialData(int *, int *, int*, int);
//int gettimeofday(struct timeval *, struct timezone *);
void MatrixMulOnCPU(const int *,const int *, int *,const int,const int,const int);


// Main function
int main(int argc, char **argv)
{
    printf("\n*****************************************************************************************\n");
	printf("\n%s Starting...\n", argv[0]);

	// result[M][S] = a[M][N] * b[N][S]
    int M = 1024, N = 1024, S = 1024;

	// Allocate the space of matrices A and B
	int *a = (int*)malloc(M * N * sizeof(int));
	if (NULL == a)
	{
		printf("the malloc of Matrix a is failed!\n");
		return 0;
    }
	int *b = (int*)malloc(N * S * sizeof(int));
	if (NULL == b)
	{
		printf("the malloc of Matrix b is failed!\n");
		return 0;
    }
    
    // Allocate the space of matrices 
	//Store the results of CPU and GPU
	int *c = (int *)malloc(M * S * sizeof(int));
	if (NULL == c)
	{
		printf("the malloc of Matrix C is failed!\n");
		return 0;
    }
	
	// Initialize the matrix and print them 
	initialData(a, b, c, 1024*1024);

	// The time of CPU execution
	
    struct timeval start;
	struct timeval finish;
	double totalTime = 0.0;

	//CPU execution
	gettimeofday(&start, NULL);

	MatrixMulOnCPU(a, b, c, M, N, S);

	gettimeofday(&finish, NULL);

	totalTime = (finish.tv_sec - start.tv_sec)*1000 + (finish.tv_usec - start.tv_usec)/1000;
	printf("\nThe CPU execution time is %lf ms.\n", totalTime);

    
	// Print the result of C
	printf("\n");
    printf("\nThe result of matrix C: \n");
    for(int i = 0; i < 10; i++)  printf("%d ", c[i]);
    printf("\n");
	for(int i = 1024*1024-10; i < 1024*1024; i++)  printf("%d ", c[i]);  
    printf("\n");

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
	printf("Do matrix multiplication on CPU...");
    printf("\n");

    printf("\n");
}

// Matrix multiplication on CPU
void MatrixMulOnCPU(const int * a,const int * b, int *result,const int M,const int N,const int S)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			int index = i * S + j;
			//result[index] = 0;

			for (int k = 0; k < N; k++)
			{
				result[index] += a[i * N + k] * b[k * S + j];
			}
		}
	}
}
