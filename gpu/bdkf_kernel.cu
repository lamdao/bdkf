//--------------------------------------------------------------------------
// bdkf_kernel.cu - the CUDA kernel functions of blind deconvolution on GPU
// 
// This file defines all functions for deconvolution on GPU
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __BDKF_KERNEL_H
#define __BDKF_KERNEL_H
//---------------------------------------------------------------------------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
//---------------------------------------------------------------------------------------------------------
#include "typedefs.h"
//---------------------------------------------------------------------------------------------------------
typedef texture<uchar, cudaTextureType3D, cudaReadModeNormalizedFloat> VolTexture;
//---------------------------------------------------------------------------------------------------------
__constant__ VolInfo vdim;
//---------------------------------------------------------------------------------------------------------
VolTexture vtex;
//---------------------------------------------------------------------------------------------------------
static __device__ inline dtype kabs(dtype v)
{
	return v < 0 ? -v : v;
}
//---------------------------------------------------------------------------------------------------------
static __device__ inline int GetIndex(int x, int y, int z)
{
	return z * vdim.vp + y * vdim.vx + x;
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_calc_cshift_index(int *odx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx < vdim.vn)
	{
		int n = idx;
		int z = n / vdim.vp; n %= vdim.vp;

		int y = n / vdim.vx;
		if (y >= vdim.sy) {
			odx[idx] = -1;
			return;
		}
		int x = n % vdim.vx;

		int q = (z + vdim.sz) % vdim.vz;
		int p = (y + vdim.sy) % vdim.vy;
		int o = (x + vdim.sx) % vdim.vx;
		odx[idx] = GetIndex(o, p, q);
	}
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_cshift(dtype *dst, int *ddx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//if (idx < vdim.vn)
	{
		int odx = ddx[idx];
		if (odx < 0) {
			return;
		}
		dtype si = dst[idx];
		dst[idx] = dst[odx];
		dst[odx] = si;
	}
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_sum(dtype *v, dtype *r)
{
	__shared__ dtype sr[NUM_THREADS];
#define t threadIdx.x
	sr[t] = v[t];
	__syncthreads();

	for (int s = NUM_THREADS/2; s > 32; s >>= 1) {
		if (t < s) {
			sr[t] += sr[t+s];
		}
		__syncthreads();
	}

	if (t < 32) {
		sr[t] += sr[t+32];
		sr[t] += sr[t+16];
		sr[t] += sr[t+ 8];
		sr[t] += sr[t+ 4];
		sr[t] += sr[t+ 2];
		sr[t] += sr[t+ 1];
	}

	if (t == 0) {
		r[blockIdx.x] = sr[0];
	}
#undef t
}
//---------------------------------------------------------------------------------------------------------
static __device__ inline cufftComplex cn_mul(const cufftComplex &a, const cufftComplex &b)
{
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}
//---------------------------------------------------------------------------------------------------------
static __device__ inline cufftComplex cj_mul(const cufftComplex &a, const cufftComplex &b)
{
	cufftComplex c;
	c.x = a.x * b.x + a.y * b.y;
	c.y = a.y * b.x - a.x * b.y;
	return c;
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_nconvolve(cufftComplex *ca, cufftComplex *cb)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < vdim.vc) {
		ca[n] = cn_mul(ca[n], cb[n]);
	}
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_cconvolve(cufftComplex *ca, cufftComplex *cb)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < vdim.vc) {
		ca[n] = cj_mul(ca[n], cb[n]);
	}
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_divfilter(dtype *v, dtype *t)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	dtype d = t[n] / vdim.vn;
	t[n] = d <= (dtype)1E-12 ? (dtype)0.0 : v[n]/d;
}
//---------------------------------------------------------------------------------------------------------
//#define USE_UNROLL_LAST_WARP
__global__
void d_dcvupdate(dtype *dv, dtype *dt, dtype *pv, dtype *dr, dtype nv, dtype gv, dtype vg)
{
	__shared__ dtype sr[NUM_THREADS], ss[NUM_THREADS];

	int t = threadIdx.x;
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	dtype v = dv[n];
	dtype d = v * dt[n] / vdim.vn;
	dtype k = pv[n] / (pv[n] + nv);
	pv[n] = pv[n] * (dtype)(1.0 - k);
	d = gv * v + vg * d + k * (d - v);
	if (d < 0) d = (dtype)0;
	sr[t] = fabsf(v - d);
	ss[t] = d;
	dv[n] = d;

	__syncthreads();

#ifndef USE_UNROLL_LAST_WARP
	for (int s = NUM_THREADS/2; s > 2; s >>= 1) {
		if (t < s) {
			sr[t] += sr[t+s];
			ss[t] += ss[t+s];
		}
		__syncthreads();
	}
#else
	for (int s = NUM_THREADS/2; s > 32; s >>= 1) {
		if (t < s) {
			sr[t] += sr[t+s];
			ss[t] += ss[t+s];
		}
		__syncthreads();
	}

	if (t < 32) {
		sr[t] += sr[t+32];
		sr[t] += sr[t+16];
		sr[t] += sr[t+ 8];
		sr[t] += sr[t+ 4];
		sr[t] += sr[t+ 2];
		sr[t] += sr[t+ 1];
		ss[t] += ss[t+32];
		ss[t] += ss[t+16];
		ss[t] += ss[t+ 8];
		ss[t] += ss[t+ 4];
		ss[t] += ss[t+ 2];
		ss[t] += ss[t+ 1];
	}
#endif
	if (t == 0) {
		dr[blockIdx.x] = sr[0]+sr[1]+sr[2]+sr[3];
		dr[blockIdx.x+gridDim.x] = ss[0]+ss[1]+ss[2]+ss[3];
	}
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_fillmap(bool *bmp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < vdim.vn) {
		int mx = vdim.vx - 2;
		int my = vdim.vy - 2;
		int mz = vdim.vz - 2;

		int n = idx;
		int z = (int)(n / vdim.vp); n %= vdim.vp;
		int y = (int)(n / vdim.vx);
		int x = (int)(n % vdim.vx);
		bmp[idx] = x > 0 && x <= mx && y > 0 && y <= my && z > 0 && z <= mz;
	}
}
//---------------------------------------------------------------------------------------------------------
static __device__ inline int d_mirror(int r, int limit)
{
	if (r >= limit) return (limit << 1) - r - 1;
	if (r < 0) return -r;
	return r;
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_crop(dtype *bsrc, dtype *bdst, int3 bp, int3 lim)
{
#define	p	vdim.vp
#define	w	vdim.vx
	int n = blockIdx.x * blockDim.x + threadIdx.x;
//	if (n < vdim.vn) {
		int k = n;
		int z = d_mirror(bp.z + (k / p), lim.z); k %= p;
		int y = d_mirror(bp.y + (k / w), lim.y);
		int x = d_mirror(bp.x + (k % w), lim.x);
		bsrc[n] = bdst[n] = tex3D(vtex, x, y, z);
//	}
#undef w
#undef p
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_calc_store_index(int *didx, int *sidx, cuVdim rdim, int sofs, cuVdim cdim)
{
#define	w	cdim.vx
#define p	cdim.vp
#define s	cdim.vn
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < s) {
		int k = n;
		int z = k / p; k %= p;
		int y = k / w;
		int x = k % w;
		sidx[n] = sofs + z * vdim.vp + y * vdim.vx + x; 
		didx[n] = z * rdim.vp + y * rdim.vx + x;
	}
#undef s
#undef p
#undef w
}
//---------------------------------------------------------------------------------------------------------
__global__
void d_store(dtype *dst, int dofs, int *didx, dtype *src, int *sidx, int s)
{
	__shared__ dtype ss[NUM_THREADS];
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	ss[threadIdx.x] = src[sidx[n]];
	__syncthreads();
//	if (n < s) {
	dst[dofs + didx[n]] = ss[threadIdx.x];
//	}
}
//---------------------------------------------------------------------------------------------------------
#endif
