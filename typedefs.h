//--------------------------------------------------------------------------
// typedef.h - commonly used data type/structure used in bdkf_main.cpp
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __TYPE_DEFS_H
#define __TYPE_DEFS_H
//--------------------------------------------------------------------------
#include <unistd.h>
#include <signal.h>
//--------------------------------------------------------------------------
#ifndef dtype
#define mk_dtype3	make_float3
#define dtype	float
#define dtype3	float3
#define dtype4	float4
#endif
//--------------------------------------------------------------------------
#ifndef ushort
#define ushort unsigned short
#endif
#ifndef uint
#define uint unsigned int
#endif
#ifndef uchar
#define uchar unsigned char
#endif
//--------------------------------------------------------------------------
typedef struct {
	int vx, vy, vz;
	int sx, sy, sz;
	int vp, vn, vc;
} VolInfo;
//--------------------------------------------------------------------------
typedef struct {
	int vx, vy, vz, vp, vn;
} cuVdim;
//--------------------------------------------------------------------------
#endif
