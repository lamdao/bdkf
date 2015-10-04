//--------------------------------------------------------------------------
// timer.cpp - elapsed time handling
// 
// This file provides functions for handling calculation elapsed time
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include "timer.h"
//--------------------------------------------------------------------------
namespace timer {
//--------------------------------------------------------------------------
static inline timeval Init()
{
	timeval t;
	gettimeofday(&t, 0);
	return t;
}
//--------------------------------------------------------------------------
static timeval t0 = Init();
static timeval t1;
//--------------------------------------------------------------------------
void Start(timeval *t0)
{
	gettimeofday(t0 ? t0 : &t1, 0);
}
//--------------------------------------------------------------------------
float Check(timeval *t0)
{
	timeval t;
	gettimeofday(&t, 0);
	if (t0) {
		t.tv_sec  -= t0->tv_sec;
		t.tv_usec -= t0->tv_usec;
	} else {
		t.tv_sec  -= t1.tv_sec;
		t.tv_usec -= t1.tv_usec;
	}
	while (t.tv_usec < 0) {
		t.tv_sec  -= 1;
		t.tv_usec += 1000000;
	}
	return t.tv_sec + 0.000001 * t.tv_usec;
}
//--------------------------------------------------------------------------
float GetRuntime()
{
	return Check(&t0);
}
//--------------------------------------------------------------------------
} // namespace Timer
