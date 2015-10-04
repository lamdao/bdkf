//--------------------------------------------------------------------------
// options.h - command line parameters parsing utilities header file
// 
// This file provides interfaces for command parameters parsing functions.
// It also declares some common variables that are used in the main
// processing file bdkf_main.cpp
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __DCV_OPTIONS_H
#define __DCV_OPTIONS_H
//--------------------------------------------------------------------------
#include "typedefs.h"
//--------------------------------------------------------------------------
extern bool verbose;
//--------------------------------------------------------------------------
extern int niters;
extern int bsize;
//--------------------------------------------------------------------------
extern dtype vstop;
extern dtype kstop;
extern int devNo;
//--------------------------------------------------------------------------
extern const char *vsource;
extern const char *dsource;
extern const char *voutput;
//--------------------------------------------------------------------------
void GetOptions(int n, char **params);
//--------------------------------------------------------------------------
#endif
