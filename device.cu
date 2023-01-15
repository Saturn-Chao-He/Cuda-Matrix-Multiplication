//******************************************************************************
// Fall 2020
// Assignment: device.cu
// Instructor: Dr. Jiang
// Programmer: Chao He
//******************************************************************************

#include <cuda_runtime.h>
#include <stdio.h>
// common header
#include "common.h" 

int main(int argc, char **argv)
{
    printf("\n%s Starting...\n", argv[0]);

    // Display the number of GPU
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess){
           //printf("cudaGetDeviceCount retruned %d \n-> %s \n"), (int)error_id, cudaGetErrorString(error_id);
           printf("Result = FALL\n");
           exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("\nDetected %d CUDA Capable device(s)\n\n", deviceCount);
    }

    // cudaDeviceProp struct
// Device 0
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // Display the properties that the slides mentioned 
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    printf("*****************************************************************************************************************\n");

    printf("  Total amount of global memory:                 %.2f GBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    
    printf("  Total amount of shared memory per block:       %lu bytes\n",
    deviceProp.sharedMemPerBlock);

    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);

    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);

    printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp.memPitch);
    
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);

    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);

    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);

    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);

    printf("  Texture Alignment:                             %lu bytes\n", deviceProp.textureAlignment);

    printf("  Can the device concurrently copy memory while executing kernels? ");
    if (deviceProp.deviceOverlap == 1 ){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }

    printf("  The Number of multiprocessors available on device: %d\n", deviceProp.multiProcessorCount);

    printf("  The Runtime limit:  ");
    if (deviceProp.kernelExecTimeoutEnabled == 1){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }
    printf("  Integrated or not:  ");
    if (deviceProp.integrated == 1){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }

    printf("  Can map host memory into the CUDA device or not:  ");
    if (deviceProp.canMapHostMemory == 1){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }

    printf("  Compute Mode:  %d\n", deviceProp.computeMode);

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);

   // printf("  Max Dimensions for 2D texture arrays (%d,%d,%d)\n", 
    //       deviceProp.maxTexture2DArray[0], deviceProp.maxTexture2DArray[1],
    //       deviceProp.maxTexture2DArray[2]);

    printf("  Whether the device supports executing multiple kernels within the same context simultanously or not:  ");
           if (deviceProp.concurrentKernels == 1){
               printf("Yes.\n");
           }else{
               printf("No.\n");
           }
    
    // Extra properties code from book
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp.l2CacheSize);
    }

    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);

// Device 1 **********************************************************************************************
    int dev1 = 1, driverVersion1 = 0, runtimeVersion1 = 0;
    CHECK(cudaSetDevice(dev1));
    cudaDeviceProp deviceProp1;
    CHECK(cudaGetDeviceProperties(&deviceProp1, dev1));

    // Display the properties that the slides mentioned 
    printf("\n\nDevice %d: \"%s\"\n", dev1, deviceProp1.name);

    printf("*****************************************************************************************************************\n");


    printf("  Total amount of global memory:                 %.2f GBytes (%llu "
           "bytes)\n", (float)deviceProp1.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp1.totalGlobalMem);
    
    printf("  Total amount of shared memory per block:       %lu bytes\n",
    deviceProp1.sharedMemPerBlock);

    printf("  Total number of registers available per block: %d\n",
           deviceProp1.regsPerBlock);

    printf("  Warp size:                                     %d\n",
           deviceProp1.warpSize);

    printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp1.memPitch);
    
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp1.maxThreadsPerBlock);
    
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp1.maxThreadsDim[0],
           deviceProp1.maxThreadsDim[1],
           deviceProp1.maxThreadsDim[2]);

    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp1.maxGridSize[0],
           deviceProp1.maxGridSize[1],
           deviceProp1.maxGridSize[2]);

    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp1.totalConstMem);

    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp1.major, deviceProp1.minor);

    printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp1.clockRate * 1e-3f,
           deviceProp1.clockRate * 1e-6f);

    printf("  Texture Alignment:                             %lu bytes\n", deviceProp1.textureAlignment);

    printf("  Can the device concurrently copy memory while executing kernels? ");
    if (deviceProp1.deviceOverlap == 1 ){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }

    printf("  The Number of multiprocessors available on device: %d\n", deviceProp1.multiProcessorCount);

    printf("  The Runtime limit:  ");
    if (deviceProp1.kernelExecTimeoutEnabled == 1){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }
    printf("  Integrated or not:  ");
    if (deviceProp1.integrated == 1){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }

    printf("  Can map host memory into the CUDA device or not:  ");
    if (deviceProp1.canMapHostMemory == 1){
        printf("Yes.\n");
    }else{
        printf("No.\n");
    }

    printf("  Compute Mode:  %d\n", deviceProp1.computeMode);

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp1.maxTexture1D,
           deviceProp1.maxTexture2D[0], deviceProp1.maxTexture2D[1],
           deviceProp1.maxTexture3D[0], deviceProp1.maxTexture3D[1],
           deviceProp1.maxTexture3D[2]);

   // printf("  Max Dimensions for 2D texture arrays (%d,%d,%d)\n", 
    //       deviceProp1.maxTexture2DArray[0], deviceProp1.maxTexture2DArray[1],
    //       deviceProp1.maxTexture2DArray[2]);

    printf("  Whether the device supports executing multiple kernels within the same context simultanously or not:  ");
           if (deviceProp1.concurrentKernels == 1){
               printf("Yes.\n");
           }else{
               printf("No.\n");
           }
    
    // Extra properties code from book
    cudaDriverGetVersion(&driverVersion1);
    cudaRuntimeGetVersion(&runtimeVersion1);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion1 / 1000, (driverVersion1 % 100) / 10,
           runtimeVersion1 / 1000, (runtimeVersion1 % 100) / 10);

    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp1.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp1.memoryBusWidth);

    if (deviceProp1.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp1.l2CacheSize);
    }

    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp1.maxTexture1DLayered[0],
           deviceProp1.maxTexture1DLayered[1], deviceProp1.maxTexture2DLayered[0],
           deviceProp1.maxTexture2DLayered[1],
           deviceProp1.maxTexture2DLayered[2]);
    
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp1.maxThreadsPerMultiProcessor);
    
    exit(EXIT_SUCCESS);


    return 0;
}
