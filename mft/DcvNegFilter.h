//--------------------------------------------------------------------------
// DcvNegFilter.h - negative value filter
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __DCV_NEGFILTER_H
#define __DCV_NEGFILTER_H
//--------------------------------------------------------------------------
#include "Resources.h"
//--------------------------------------------------------------------------
class DcvNegFilter: public Resources
{
public:
	dtype *Execute(dtype *dv, dtype *dt)
	{
		for (size_t n = 0; n < vn; n++) {
			dtype d = dt[n] / vn;
			dt[n] = (d <= 1E-12 ? 0.0f : dv[n]/d);
		}
		return dt;
	}
};
//--------------------------------------------------------------------------
#endif
