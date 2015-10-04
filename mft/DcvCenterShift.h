//--------------------------------------------------------------------------
// DcvCenterShift.h - 3D center shift required before FFT to compute OTF
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __DCV_CENTERSHIFT_H
#define __DCV_CENTERSHIFT_H
//--------------------------------------------------------------------------
#include "Resources.h"
//--------------------------------------------------------------------------
class DcvCenterShift: public Resources
{
public:
	dtype *Execute(dtype *src, dtype *dst)
	{
		if (dst != src) {
			memcpy(dst, src, vn * sizeof(dtype));
		}

		for (size_t idx = 0; idx < vn; idx++) {
			int x, y, z;
			GetPosition(idx, x, y, z);
			if (y >= cy) continue;

			int q = (z + cz) % vz;
			int p = (y + cy) % vy;
			int o = (x + cx) % vx;
			size_t odx = GetIndex(o, p, q);

			dtype v = src[idx];
			dst[idx] = src[odx];
			dst[odx] = v;
		}
		return dst;
	}
};
//--------------------------------------------------------------------------
#endif
