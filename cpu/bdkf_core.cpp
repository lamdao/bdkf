//--------------------------------------------------------------------------
// bdkf_core.cpp - the core of blind deconvolution
// 
// This file defines all functions for handling memory, cropping/merging
// sub-volumes, and deconvolve a volume
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <fftw3.h>
//--------------------------------------------------------------------------
#include "options.h"
#include "volume.h"
#include "point.h"
//--------------------------------------------------------------------------
#include "DcvCenterShift.h"
#include "DcvConvolver.h"
#include "DcvNegFilter.h"
#include "DcvUpdater.h"
//--------------------------------------------------------------------------
namespace bdkf {
//--------------------------------------------------------------------------
namespace Device {
	void Init() { dcv_init_mthreads(); }
	void Release() { dcv_cleanup(); }
}
//--------------------------------------------------------------------------
// Blind Solver
//--------------------------------------------------------------------------
class Solver: public Resources
{
public:
	static dtype *df0;
	static Volume<dtype> vol;
	static Volume<dtype> out;
	static Volume<dtype> dfi;
private:
	static dtype *t;
	static ctype *h, *y;
	static dtype *v, *o;
	static dtype *k;

	static DcvCenterShift cshift;
	static DcvConvolver conv;
	static DcvNegFilter nftr;
	static DcvUpdater vu, ku;

	static dcv_plan fwd_k2h;
	static dcv_plan fwd_t2h;
	static dcv_plan bwd_h2t;
					
	static dcv_plan fwd_o2y;
	static dcv_plan fwd_t2y;
	static dcv_plan bwd_y2t;
public:
	static void Init(const Dim &wbs) {
		t = dcv_ralloc(vn);
		h = dcv_calloc(vc);
		y = dcv_calloc(vc);

		vol.Recreate(wbs); v = vol;
		out.Recreate(wbs); o = out;
		dfi.Recreate(wbs); k = dfi;

		vu.Init(o, t);
		ku.Init(k, t);

		dcv_plan_mthreads();

		fwd_k2h = dcv_r3d(k, h);
		fwd_t2h = dcv_r3d(t, h);
		bwd_h2t = dcv_c3d(h, t);

		fwd_o2y = dcv_r3d(o, y);
		fwd_t2y = dcv_r3d(t, y);
		bwd_y2t = dcv_c3d(y, t);
	}
	static void Release() {
		dcv_close(fwd_o2y);
		dcv_close(fwd_t2y);
		dcv_close(bwd_y2t);

		dcv_close(fwd_k2h);
		dcv_close(fwd_t2h);
		dcv_close(bwd_h2t);

		dcv_free(y);
		dcv_free(h);
		dcv_free(t);
	}
	static void Reset() {
		vu.Reset();
		ku.Reset();
		memcpy(dfi, df0, dfi.Size());
	}
	static void Prepare(double &sv, double &sk) {
		sv = sk = 0;
		memcpy(o, v, vn * sizeof(dtype));
		for (int i = 0; i < vn; i++) {
			sv += v[i];
		}
		for (int i = 0; i < vn; i++) {
			sk += k[i];
		}
	}
	static int Run(dtype &rv, dtype &rk) {
		int ni;
		double sv, sk;
		bool kstable = false;

		Reset();
		Prepare(sv, sk);
		for (rk = 1.0, ni = 0; ni < niters; ni++) {
			if (!kstable) {
				cshift.Execute(k, t);			// h = otf(k)
				dcv_fft(t2h);					//   = fft(t = cshift(k))
			}
			dcv_fft(o2y);						// y = fft(o)
			y = conv.Execute<NORM>(y, h);		// y = convol(y, h)
			dcv_bft(y2t);						// t = ifft(y)
			t = nftr.Execute(v, t);				// t = neg_filter(v, t)
			dcv_fft(t2y);						// y = fft(t)
			y = conv.Execute<CONJ>(y, h);		// y = convol(y, conj(h))
			dcv_bft(y2t);						// t = ifft(y)
			rv = vu.Execute(sv);				// o = update(o, t, d)
			if (rv <= vstop)
				break;
			if (rk <= kstop)
				continue;
			cshift.Execute(o, t);				// y = otf(o)
			dcv_fft(t2y);						//   = fft(t = cshift(o))
			dcv_fft(k2h);						// h = fft(k)
			h = conv.Execute<NORM>(h, y);		// h = convol(h, y)
			dcv_bft(h2t);						// t = ifft(h)
			t = nftr.Execute(v, t);				// t = neg_filter(v, t)
			dcv_fft(t2h);						// h = fft(t)
			h = conv.Execute<CONJ>(h, y);		// h = convol(h, conj(y))
			dcv_bft(h2t);						// t = ifft(h)
			rk = ku.Execute(sk);				// k = update(k, t, d)
			if (rk <= kstop) {
				kstable = true;
				cshift.Execute(k, t);
				dcv_fft(t2h);
			}
		}
		return ni;
	}
};
//--------------------------------------------------------------------------
dtype *Solver::df0;
//--------------------------------------------------------------------------
Volume<dtype> Solver::vol;
Volume<dtype> Solver::out;
Volume<dtype> Solver::dfi;
//--------------------------------------------------------------------------
dtype *Solver::t;
ctype *Solver::h;
ctype *Solver::y;
dtype *Solver::v;
dtype *Solver::o;
dtype *Solver::k;
//--------------------------------------------------------------------------
DcvCenterShift Solver::cshift;
DcvConvolver Solver::conv;
DcvNegFilter Solver::nftr;
DcvUpdater Solver::vu;
DcvUpdater Solver::ku;
//--------------------------------------------------------------------------
dcv_plan Solver::fwd_k2h;
dcv_plan Solver::fwd_t2h;
dcv_plan Solver::bwd_h2t;
//--------------------------------------------------------------------------
dcv_plan Solver::fwd_o2y;
dcv_plan Solver::fwd_t2y;
dcv_plan Solver::bwd_y2t;
//--------------------------------------------------------------------------
// bdkf interfaces
//--------------------------------------------------------------------------
Dim roi;
//--------------------------------------------------------------------------
Volume<uchar> src;
Volume<dtype> res;
//--------------------------------------------------------------------------
void Init(const Dim &wbs, const Dim &cbs)
{
	roi = cbs;

	Resources::Init(wbs.x, wbs.y, wbs.z);
	Solver::Init(wbs);

	size_t vp = (size_t)wbs.x * wbs.y;
	size_t vn = vp * wbs.z;

	printf("- Block size: %dx%dx%d\n",
			cbs.x, cbs.y, cbs.z);
	printf("- Working size: %dx%dx%d\n"
			"   < %d voxels/plane, %d voxels/volume >\n",
			wbs.x, wbs.y, wbs.z, vp, vn);
}
//--------------------------------------------------------------------------
void InitSourceBuffer(void *sb, const Dim &dim)
{
	src.Recreate(dim, sb);
}
//--------------------------------------------------------------------------
void InitDF0(void *k0)
{
	Solver::df0 = (dtype *)k0;
}
//--------------------------------------------------------------------------
void CreateResultBuffer()
{
	res.Recreate(src.dim);
}
//--------------------------------------------------------------------------
static inline int dmirror(int r, int limit)
{
	if (r >= limit)
		return (limit << 1) - r - 1;
	if (r < 0)
		return -r;
	return r;
}
//--------------------------------------------------------------------------
// Crop from [src] at [bp] to [vol]
void CropSourceBlock(const Pos &bp)
{
	dtype *v = Solver::vol;
	Dim dim = Solver::vol.dim;
	for (int z = 0; z < dim.z; z++) {
		int zi = dmirror(bp.z + z, src.dim.z);
		for (int y = 0; y < dim.y; y++) {
			int yi = dmirror(bp.y + y, src.dim.y);
			for (int x = 0; x < dim.x; x++) {
				int xi = dmirror(bp.x + x, src.dim.x);
				*v++ = (dtype)src(xi, yi, zi);
			}
		}
	}
}
//--------------------------------------------------------------------------
// Put [out[center::roi]] to [res] at [bp]
void SaveResultBlock(const Pos &bp)
{
	Pos sp = (Solver::out.dim - roi) / 2;
	for (int z = 0; z < roi.z; z++) {
		int dz = bp.z + z;
		int sz = sp.z + z;
		for (int y = 0; y < roi.y; y++) {
			int dy = bp.y + y;
			int sy = sp.y + y;
			for (int x = 0; x < roi.x; x++) {
				int dx = bp.x + x;
				int sx = sp.x + x;
				res(dx, dy, dz) = Solver::out(sx, sy, sz);
			}
		}
	}
}
//--------------------------------------------------------------------------
dtype *GetCurrentDF()
{
	return Solver::dfi;
}
//--------------------------------------------------------------------------
dtype *GetFinalResult()
{
	src.Detach();
	return res;
}
//--------------------------------------------------------------------------
int Exec(dtype &rv, dtype &rk)
{
	Solver::Run(rv, rk);
}
//--------------------------------------------------------------------------
void Done()
{
	Solver::Release();
	Resources::Release();
	Device::Release();
}
//--------------------------------------------------------------------------
} // namespace bdkf
