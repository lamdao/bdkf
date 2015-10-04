//------------------------------------------------------------------------
// kuda_utils.cpp - GPU utility functions
//------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//------------------------------------------------------------------------
#include "kuda_utils.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//------------------------------------------------------------------------
void PrintCudaError(const char *fx, int n)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("[%s @ %d] CUDA error: %s.", fx, n, cudaGetErrorString(err));
	}
}
//------------------------------------------------------------------------
bool CudaInit(int devNo)
{
	int n, v;

	if (cudaGetDeviceCount(&n) != cudaSuccess) {
		printf("Get dev count error!!!!\n", 0);
		return false;
	}

	if (devNo >= n) {
		printf("Cannot set Cuda Device #%d.\n", devNo);
		return false;
	}

	cudaSetDevice(devNo);
	if (cudaDriverGetVersion(&v) != cudaSuccess)
		return false;

	printf("* CUDA:\n  o Toolkit/Driver version: %d / %d\n", CUDA_VERSION, v);
	if ((v / 1000) < (CUDA_VERSION/1000)) {
		return false;
	}
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devNo);
	printf("  o Device name: %s.\n", prop.name);
	if (!prop.canMapHostMemory) {
		printf("Cannot use Host memory map.\n", 0);
	} else {
		cudaSetDeviceFlags(cudaDeviceMapHost);
	}
	#define S	"    "
	printf(S"- GPU integrated: %s.\n", prop.integrated ? "Yes" : "No");
	printf(S"- WarpSize = %d\n", prop.warpSize);
	printf(S"- SharedMemPerBlock = %d\n", prop.sharedMemPerBlock);
	printf(S"- MaxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
	printf(S"- MaxGridSize = %d %d %d\n",
			prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf(S"- MultiProcessorCount = %d\n", prop.multiProcessorCount);
	printf(S"- MaxThreadsPerMultiProcessor = %d\n\n",
			prop.maxThreadsPerMultiProcessor);

	return n > 0;
}
//------------------------------------------------------------------------
