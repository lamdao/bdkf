//--------------------------------------------------------------------------
// bdkf_core.h - common interface of CPU & GPU core implementation of 
//               3D blind deconvolution
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __BDKF_CORE_H
#define __BDKF_CORE_H
//--------------------------------------------------------------------------
#include "typedefs.h"
#include "point.h"
//--------------------------------------------------------------------------
namespace bdkf
{
namespace Device {
void Init();
void Release();
}
void Init(const Dim &wbs, const Dim &cbs);
void InitSourceBuffer(void *sb, const Dim &dim);
void InitDF0(void *k0);
void CreateResultBuffer();
void CropSourceBlock(const Pos &bp);
int Exec(dtype &rv, dtype &rk);
void SaveResultBlock(const Pos &bp);
dtype *GetCurrentDF();
dtype *GetFinalResult();
void Done();
}
//--------------------------------------------------------------------------
#endif
