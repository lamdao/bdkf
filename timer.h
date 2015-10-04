//--------------------------------------------------------------------------
// timer.cpp - elapsed time handling header file
// 
// This file declares functions for handling calculation elapsed time
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __DCV_TIMER_H
#define __DCV_TIMER_H
//--------------------------------------------------------------------------
#include <sys/time.h>
//--------------------------------------------------------------------------
namespace timer
{
void Start(timeval *t0 = 0);
float Check(timeval *t0 = 0);
float GetRuntime();
}
//--------------------------------------------------------------------------
#endif
