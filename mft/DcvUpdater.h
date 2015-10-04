//--------------------------------------------------------------------------
// DcvUpdater.h - Update the volume with gradient adjustment
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __DCV_UPDATER_H
#define __DCV_UPDATER_H
//--------------------------------------------------------------------------
#include "Resources.h"
//--------------------------------------------------------------------------
#define nv  (dtype)Resources::noisevar
#define gv  (dtype)Resources::gain
//--------------------------------------------------------------------------
enum {
	FT_NONE,
	FT_KALMAN,	// Kalman
	FT_TV		// Total Variation
};
//--------------------------------------------------------------------------
class DcvUpdater: public Resources
{
private:
	dtype *dv;
	dtype *dt;
	std::vector<dtype> pv;
public:
//	DcvUpdater(): pv() {}

	void Init(dtype *vol, dtype *tmp)
	{
		dv = vol;
		dt = tmp;
		pv = std::vector<dtype>(vn);
	}

	void Reset()
	{
		for (int n = 0; n < vn; n++)
			pv[n] = nv;
	}

	template<int filter = FT_KALMAN>
	double Execute(double &sum)
	{
		double rcv = 0, scv = 0;
		dtype vg = (dtype)(1.0 - gv);
		for (size_t n = 0; n < vn; n++) {
			dtype d = dv[n] * dt[n] / vn;	// d = o
			if (!isfinite(d)) d = 0;

			if (filter == FT_KALMAN) {
				dtype k = pv[n] / (pv[n] + nv);
				pv[n] = pv[n] * (dtype)(1.0 - k);
				d = gv * dv[n] + vg * d + k * (d - dv[n]);
			}
			if (filter == FT_TV) {
			}

			if (d < 0) d = 0;
			rcv += fabs(dv[n] - d);
			dv[n] = d;
			scv += d;
		}

		rcv = sum != 0.0 ? rcv / sum : 0.0;
		sum = scv;

		return rcv;
	}
};
//--------------------------------------------------------------------------
#endif
