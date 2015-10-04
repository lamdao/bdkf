//--------------------------------------------------------------------------
// bdkf_core.cu - the CUDA core of blind deconvolution on GPU
// 
// This file defines all functions for handling memory, cropping/merging
// sub-volumes, and deconvolve sub-volume
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include "bdkf_kernel.cu"
#include "kuda_utils.h"
#include "options.h"
#include "point.h"
//--------------------------------------------------------------------------
namespace bdkf {
//--------------------------------------------------------------------------
namespace Device {
	void Init()
	{
		extern int devNo;
		// Initialize GPU with selected GPU id
		// - Default devNo = 0
		if (!CudaInit(::devNo))
			exit(0);
	}
	void Release() {}
}
//--------------------------------------------------------------------------
Dim wbs(0,0,0);
Dim cbs(0,0,0);
//--------------------------------------------------------------------------
dtype pgain = (dtype)0.5;
dtype nweight = (dtype)0.05;
//--------------------------------------------------------------------------
//#define USE_VERBOSE_PRINT
//--------------------------------------------------------------------------
static inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
//--------------------------------------------------------------------------
cufftHandle plan_r2c;
cufftHandle plan_c2r;
//--------------------------------------------------------------------------
dtype *d_src, *d_dst, *d_ker, *d_t, *d_df0;
//--------------------------------------------------------------------------
cufftComplex *d_y, *d_h;
//--------------------------------------------------------------------------
dtype *vdiv;
dtype *nweights;
//--------------------------------------------------------------------------
dtype *d_tmp, *h_tmp;
dtype *d_result, *h_result;
//--------------------------------------------------------------------------
int *d_odx;
int *d_dx, *d_sx;
//--------------------------------------------------------------------------
cudaArray *d_volArray = 0;
//--------------------------------------------------------------------------
int VX, VY, VZ;
//--------------------------------------------------------------------------
int VP, VN, VS;
//--------------------------------------------------------------------------
#define VT	NUM_THREADS * sizeof(dtype)
#define VB	NUM_BLOCKS * sizeof(dtype)
//--------------------------------------------------------------------------
int NUM_BLOCKS = 1;
int NUM_STORE_BLOCKS;
//--------------------------------------------------------------------------
#ifdef USE_VERBOSE_PRINT
#define vbprintf(fmt,args...)	if (verbose) printf(fmt,args)
#define nbprintf(fmt,args...)	if (!verbose) printf(fmt,args)
#else
#define vbprintf(fmt,args...)
#define nbprintf(fmt,args...)
#endif
//--------------------------------------------------------------------------
#define d_exec(fx)	d_##fx<<<NUM_BLOCKS, NUM_THREADS>>>
//--------------------------------------------------------------------------
void Init(const Dim &wbs, const Dim &cbs)
{
	VolInfo vi;

	vi.vx = VX = wbs.x; vi.sx = VX / 2;
	vi.vy = VY = wbs.y; vi.sy = VY / 2;
	vi.vz = VZ = wbs.z; vi.sz = VZ / 2;
	vi.vp = VP = VX * VY;
	vi.vn = VN = VP * VZ;
	vi.vc = (VX/2+1) * VY * VZ;
	VS = VN * sizeof(dtype);
	NUM_BLOCKS = iDivUp((int)VN, NUM_THREADS);
	NUM_STORE_BLOCKS = (bsize * bsize * bsize) / NUM_THREADS;

	printf("- Block size: %dx%dx%d\n",
			cbs.x, cbs.y, cbs.z);
	printf("- Working size: %dx%dx%d\n"
			"   < %d voxels/plane, %d voxels/volume >\n",
			VX, VY, VZ, VP, VN);
	printf("- GPU settings:\n"
			"   < NumBlocks = %d, NumThreads = %d >\n",
			NUM_BLOCKS, NUM_THREADS);

	cudaMemcpyToSymbol(vdim, &vi, sizeof(VolInfo));

	nweights = new dtype[VS];
	for (int i = 0; i < VS; i++)
		nweights[i] = (dtype)nweight;
	kudaAlloc(vdiv, VS);

	kudaAlloc(d_src, VS);
	kudaAlloc(d_dst, VS);
	kudaAlloc(d_ker, VS);
	kudaAlloc(d_df0, VS);
	kudaAlloc(d_tmp, VS); d_t = d_tmp;
	kudaAlloc(d_odx, VN * sizeof(int));

	kudaAlloc(d_y, sizeof(cufftComplex)*vi.vc);
	kudaAlloc(d_h, sizeof(cufftComplex)*vi.vc);

	if (CUFFT_SUCCESS != cufftPlan3d(&plan_r2c, VX, VY, VZ, CUFFT_R2C))
		kudaShowErrorExit("Create plan r2c");
	if (CUFFT_SUCCESS != cufftPlan3d(&plan_c2r, VX, VY, VZ, CUFFT_C2R))
		kudaShowErrorExit("Create plan c2r");
	cufftSetCompatibilityMode(plan_r2c, CUFFT_COMPATIBILITY_NATIVE);
	cufftSetCompatibilityMode(plan_c2r, CUFFT_COMPATIBILITY_NATIVE);

	d_exec(calc_cshift_index)(d_odx);
	kudaSync("Init cshift index error");

	cudaHostAlloc(&h_tmp, 2*VB, cudaHostAllocMapped);

	bdkf::wbs = wbs;
	bdkf::cbs = cbs;
}
//--------------------------------------------------------------------------
dtype calc_total(dtype *v)
{
	d_exec(sum)(v, d_tmp);
	kudaSync(__FUNCTION__);
	kudaMget(h_tmp, d_tmp, VB);

	dtype r = 0;
	for (int n = 0; n < NUM_BLOCKS; n++)
		r += h_tmp[n];
	return r;
}
//--------------------------------------------------------------------------
dtype dcv_update(dtype *v, dtype *t, dtype &sum)
{
	d_exec(dcvupdate)(v, t, vdiv, d_tmp,
						(dtype)nweight, (dtype)pgain, (dtype)1.0 - pgain);
	kudaSync(__FUNCTION__);
	kudaMget(h_tmp, d_tmp, 2 * VB);

	dtype rcv = 0, scv = 0;
	dtype *rt = h_tmp, *st = &h_tmp[NUM_BLOCKS];
	for (int i = 0; i < NUM_BLOCKS; i++) {
		rcv += *rt++;
		scv += *st++;
	}
	rcv = rcv / sum;
	sum = scv;
	return rcv;
}
//--------------------------------------------------------------------------
#define dcv_run(fx,a,b)									\
	d_exec(fx)(a,b);									\
	kudaSync(#fx"("#a#b")");
//--------------------------------------------------------------------------
#define dcv_ncv(a,b)	dcv_run(nconvolve, a, b)
#define dcv_ccv(a,b)	dcv_run(cconvolve, a, b)
#define dcv_dvf(a,b)	dcv_run(divfilter, a, b)
//--------------------------------------------------------------------------
#define dcv_cshift(s, d)										\
	kudaMxfr(d, s, VS);											\
	d_exec(cshift)(d, d_odx);									\
	kudaSync(#d" = cshift("#s")");
//--------------------------------------------------------------------------
#define dcv_fft(a, b)											\
	if (CUFFT_SUCCESS != cufftExecR2C(plan_r2c, a, b))			\
		kudaShowErrorExit("r2c("#a","#b")");					\
	kudaSync(#b" = fft("#a") - sync error");
//--------------------------------------------------------------------------
#define dcv_bft(a, b)											\
	if (CUFFT_SUCCESS != cufftExecC2R(plan_c2r, a, b))			\
		kudaShowErrorExit("c2r("#a","#b")");					\
	kudaSync(#b" = bft("#a") - sync error");
//--------------------------------------------------------------------------
#define o d_dst
#define v d_src
#define k d_ker
#define t d_t
#define h d_h
#define y d_y
//--------------------------------------------------------------------------
int Exec(dtype &rv, dtype &rk)
{
	int ni;
	bool kstable = false;
	dtype sv = calc_total(v), sk = calc_total(k);

	kudaMxfr(k, d_df0, VS);
	kudaMset(vdiv, nweights, VS);
	for (rk = (dtype)1.0, ni = 0; ni < niters; ni++) {
		if (!kstable) {
			dcv_cshift(k, t);				// h = otf(k)
			dcv_fft(t, h);					//   = fft(t = cshift(k))
		}
		dcv_fft(o, y);						// y = fft(o)
		dcv_ncv(y, h);						// y = y (X) h
		dcv_bft(y, t);						// t = bft(y)
		dcv_dvf(v, t);						// t = v (/) t

		dcv_fft(t, y);						// y = fft(t)
		dcv_ccv(y, h);						// y = y (X) h~
		dcv_bft(y, t);						// t = bft(y)
		rv = dcv_update(o, t, sv);			// o = kfilter(o * t)
		if (rv <= vstop)
			break;
		if (rk > kstop) {
			dcv_cshift(o, t);				// y = otf(o)
			dcv_fft(t, y);					//   = fft(t = cshift(o))
			dcv_fft(k, h);					// h = fft(k)
			dcv_ncv(h, y);					// h = h (X) y
			dcv_bft(h, t);					// t = bft(h)
			dcv_dvf(v, t);					// t = v (/) t
			dcv_fft(t, h);					// h = fft(t)
			dcv_ccv(h, y);					// h = h (X) y~
			dcv_bft(h, t);					// t = bft(h)
			rk = dcv_update(k, t, sk);		// k = kfilter(k * t)
			if (rk <= kstop) {
				kstable = true;
				dcv_cshift(k, t);			// h = otf(k)
				dcv_fft(t, h);				//   = fft(t = cshift(k))
			}
		}
		vbprintf("| #%04d: %8.6f  %8.6f\n", ni+1, rv, rk);
	}
	nbprintf("| #%04d: %8.6f  %8.6f\n", ni+1, rv, rk);
	return ni;
}
#undef v
#undef o
#undef k
#undef t
#undef h
#undef y
//--------------------------------------------------------------------------
template<class vtype>
void kudaVolAlloc(void *vol, const Dim &d, cudaArray *&d_vArray,
		texture<vtype, cudaTextureType3D, cudaReadModeNormalizedFloat> &vt)
{
	cudaExtent vdim = make_cudaExtent(d.x, d.y, d.z);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<vtype>();
	if (cudaSuccess != cudaMalloc3DArray(&d_vArray, &desc, vdim)) {
		kudaErrorExit();
	}

	cudaMemcpy3DParms copyParams = {0};
	copyParams.kind     = cudaMemcpyHostToDevice;
	copyParams.srcPtr   = make_cudaPitchedPtr(vol,
								vdim.width * sizeof(vtype),
								vdim.width, vdim.height);
	copyParams.dstArray = d_vArray;
	copyParams.extent   = vdim;
	if (cudaSuccess != cudaMemcpy3D(&copyParams)) {
		kudaErrorExit();
	}

	// access with non-normalized texture coordinates
	vt.normalized = false;
	// no interpolation 
	vt.filterMode = cudaFilterModePoint;
	// mirroring texture coordinates
	vt.addressMode[0] = cudaAddressModeBorder;
	vt.addressMode[1] = cudaAddressModeBorder;
	vt.addressMode[2] = cudaAddressModeBorder;
	// bind array to 3D texture
	if (cudaSuccess != cudaBindTextureToArray(vt, d_vArray, desc)) {
		kudaErrorExit();
	}
}
//--------------------------------------------------------------------------
Dim sdim(0,0,0);
int nVolItems = 0;
int nBlkItems = 0;
size_t rbSize = 0;
//--------------------------------------------------------------------------
inline cuVdim make_cuVdim(const Dim &d)
{
	cuVdim c;
	c.vx = d.x;
	c.vy = d.y;
	c.vz = d.z;
	c.vp = c.vx * c.vy;
	c.vn = c.vp * c.vz;
	return c;
}
//--------------------------------------------------------------------------
void CalcStoreIndex()
{
	Dim r = (wbs - cbs) / 2;
	int sofs = (r.z * wbs.y + r.y) * wbs.x + r.x;

	cuVdim cdim = make_cuVdim(cbs);
	cuVdim rdim = make_cuVdim(sdim);
	kudaAlloc(d_dx, sizeof(int) * cdim.vn);
	kudaAlloc(d_sx, sizeof(int) * cdim.vn);
	d_calc_store_index<<<NUM_STORE_BLOCKS, NUM_THREADS>>>
		(d_dx, d_sx, rdim, sofs, cdim);
	kudaSync("d_calc_store_index");

	nBlkItems = cdim.vn;
}
//--------------------------------------------------------------------------
void InitSourceBuffer(void *sb, const Dim &dim)
{
	sdim = dim;
	nVolItems = dim.x * dim.y * dim.z;
	kudaVolAlloc<uchar>(sb, dim, d_volArray, vtex);
	CalcStoreIndex();
}
//--------------------------------------------------------------------------
void InitDF0(void *k0)
{
	kudaMset(d_df0, k0, VS);
}
//--------------------------------------------------------------------------
void CreateResultBuffer()
{
	rbSize = (size_t)nVolItems * sizeof(dtype);
	kudaAlloc(d_result, rbSize);
}
//--------------------------------------------------------------------------
void CropSourceBlock(const Pos &bp)
{
	d_exec(crop)(d_src, d_dst, bp, sdim);
	kudaSync("d_crop");
}
//--------------------------------------------------------------------------
void SaveResultBlock(const Pos &bp)
{
	int dofs = (bp.z * sdim.y + bp.y) * sdim.x + bp.x;
	d_store<<<NUM_STORE_BLOCKS, NUM_THREADS>>>
		(d_result, dofs, d_dx, d_dst, d_sx, nBlkItems);
	kudaSync("d_store");
}
//--------------------------------------------------------------------------
dtype *GetCurrentDF()
{
	dtype *df = new dtype[VN];
	kudaMget(df, d_ker, VS);
	return df;
}
//--------------------------------------------------------------------------
dtype *GetFinalResult()
{
	cudaHostAlloc(&h_result, rbSize, cudaHostAllocMapped);
	ShowCudaError("cuHostAlloc");
	kudaMget(h_result, d_result, rbSize);
	cudaFree(d_result);
	return h_result;
}
//--------------------------------------------------------------------------
void Done()
{
	cudaFree(h_result);
	cudaFree(d_sx);
	cudaFree(d_dx);

	if (d_volArray != NULL) {
		cudaFreeArray(d_volArray);
		d_volArray = NULL;
	}
	cufftDestroy(plan_r2c);
	cufftDestroy(plan_c2r);

	cudaFree(d_y);
	cudaFree(d_h);
	cudaFree(d_tmp);
	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(d_ker);
	cudaFree(d_df0);
	cudaFree(vdiv);
	cudaFree(h_tmp);
	cudaFree(d_odx);

	delete [] nweights;
}
//--------------------------------------------------------------------------
} // namespace bdkf
