//------------------------------------------------------------------------
// kuda_utils.h - GPU utility functions/macros header file
//------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//------------------------------------------------------------------------
#ifndef KUDA_UTILS_H
#define KUDA_UTILS_H
//------------------------------------------------------------------------
//#define USE_KUDA_PERFORMANCE
//------------------------------------------------------------------------
bool CudaInit(int devNo);
void PrintCudaError(const char *fx="", int n=0);
//------------------------------------------------------------------------
#define ShowCudaError(msg)		PrintCudaError(msg, __LINE__)
//------------------------------------------------------------------------
// On error, show name of function that caused error and exit
#define kudaErrorExit()	{									\
		ShowCudaError(__FUNCTION__);						\
		exit(0);											\
}
//------------------------------------------------------------------------
// On error, show error message <msg> and exit
#define kudaShowErrorExit(msg)	{							\
		ShowCudaError(msg);									\
		exit(0);											\
}
//------------------------------------------------------------------------
#ifndef USE_KUDA_PERFORMANCE
//------------------------------------------------------------------------
// Synchnronize all gpu threads before next operations
#define kudaSync(msg)										\
	if (cudaSuccess != cudaThreadSynchronize()) {			\
		kudaShowErrorExit(msg);								\
	}
//------------------------------------------------------------------------
// Allocate a buffer on gpu memory
#define kudaAlloc(buf, size)								\
	if (cudaSuccess != cudaMalloc((void**)&buf, size))		\
		kudaErrorExit();
//------------------------------------------------------------------------
// Set device memory buffer to zero (0)
//   buf: gpu memory buffer
// Result:
// 	 gpu_mem<buf> = 0
#define kudaClear(buf, size)								\
	if (cudaSuccess != cudaMemset(buf, 0, size))			\
		kudaErrorExit();
//------------------------------------------------------------------------
// device-to-device memory copy
//   d: destination
//   s: source
// Result:
//   gpu_mem<d> = gpu_mem<s>
#define kudaMxfr(d, s, size)								\
	if (cudaSuccess !=										\
		cudaMemcpy(d, s, size, cudaMemcpyDeviceToDevice)) {	\
		kudaShowErrorExit("Device copy "#s" => "#d);		\
	}
//------------------------------------------------------------------------
// host-to-device memory copy
//   d: destination
//   s: source
// Result:
//   gpu_mem<d> = cpu_mem<s>
#define kudaMset(d, s, size)								\
	if (cudaSuccess !=										\
		cudaMemcpy(d, s, size, cudaMemcpyHostToDevice)) {	\
		kudaShowErrorExit("Copy to device "#s" => "#d);		\
	}
//------------------------------------------------------------------------
// device-to-host memory copy
//   d: destination
//   s: source
// Result:
//   cpu_mem<d> = gpu_mem<s>
#define kudaMget(d, s, size)								\
	if (cudaSuccess !=										\
		cudaMemcpy(d, s, size, cudaMemcpyDeviceToHost)) {	\
		kudaShowErrorExit("Copy from device "#s" => "#d);	\
	}
//------------------------------------------------------------------------
#else  // USE_KUDA_PERFORMANCE
//------------------------------------------------------------------------
// In performance compiled mode, ignore all errors (no checking)
#define kudaSync(msg)			cudaThreadSynchronize()
#define kudaAlloc(buf, size)	cudaMalloc((void**)&buf, size)
#define kudaClear(buf, size)	cudaMemset(buf, 0, size)
#define kudaMxfr(d, s, size)	cudaMemcpy(d, s, size, cudaMemcpyDeviceToDevice)
#define kudaMset(d, s, size)	cudaMemcpy(d, s, size, cudaMemcpyHostToDevice)
#define kudaMget(d, s, size)	cudaMemcpy(d, s, size, cudaMemcpyDeviceToHost)
//------------------------------------------------------------------------
#endif // if USE_KUDA_PERFORMANCE
//------------------------------------------------------------------------
#endif
