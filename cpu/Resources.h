//--------------------------------------------------------------------------
// Resources.h - Pre-calculate volume index to map position to index back
//               and forth for speedup
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __RESOURCES_H
#define __RESOURCES_H
//--------------------------------------------------------------------------
#include <vector>
//--------------------------------------------------------------------------
#include "DcvFFTW.h"
//--------------------------------------------------------------------------
class Resources
{
protected:
	static std::vector<bool> bmp;
	static std::vector<int> xts, yts, zts;

	static int mx, my, mz;
	static int vx, vy, vz;
	static int cx, cy, cz;
	static size_t vp, vn, vc;

	static double noisevar, gain;
public:
	static void Init(int w, int h, int d) {
		vx = w;
		vy = h;
		vz = d;
		vp = (size_t)vx * vy;
		vn = vp * vz;
		vc = (size_t)(vx / 2 + 1) * vy * vz;
		cx = vx / 2;
		cy = vy / 2;
		cz = vz / 2;

		Init();
	}

	static void GetPosition(size_t n, int &x, int &y, int &z) {
		x = xts[n];
		y = yts[n];
		z = zts[n];
	}

	static size_t GetIndex(int x, int y, int z) {
		return vp * z + (size_t)vx * y + x;
	}

	static void Release() {
	}

	static void FillMap();
private:
	static void Init();
};
//--------------------------------------------------------------------------
#endif
