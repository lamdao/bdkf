//--------------------------------------------------------------------------
// DcvConvolver.h - convolution operation for 2 complex sub-volume
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __DCV_CONVOLVER_H
#define __DCV_CONVOLVER_H
//--------------------------------------------------------------------------
#include "Resources.h"
//--------------------------------------------------------------------------
enum ConvolveType {
	NORM = 0,
	CONJ = 1
};
//--------------------------------------------------------------------------
class DcvConvolver: public Resources
{
public:
	template<int type>
	ctype *Execute(ctype *ca, ctype *cb)
	{
		if (type == NORM) {
			for (size_t n = 0; n < vc; n++) {
				dtype *a = ca[n];
				dtype *b = cb[n];
				dtype r = a[0] * b[0] - a[1] * b[1];
				dtype i = a[0] * b[1] + a[1] * b[0];
				a[0] = r;
				a[1] = i;
			}
		} else {
			for (size_t n = 0; n < vc; n++) {
				dtype *a = ca[n];
				dtype *b = cb[n];
				dtype r = a[0] * b[0] + a[1] * b[1];
				dtype i = a[1] * b[0] - a[0] * b[1];
				a[0] = r;
				a[1] = i;
			}
		}
		return ca;
	}
};
//--------------------------------------------------------------------------
#endif
