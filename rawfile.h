//--------------------------------------------------------------------------
// rawfile.cpp - a 3D volume in raw-format (pure data only) header file
// 
// This file declares interface for handling 3D volume in raw-format
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __RAWFILE_H
#define __RAWFILE_H
//--------------------------------------------------------------------------
#include <iostream>
#include <fstream>
//--------------------------------------------------------------------------
#include "point.h"
//--------------------------------------------------------------------------
namespace RawFile
{
Dim DimExtract(const char *filename);
void Load(const char *filename, void *data, size_t size);
char *FormatName(const char *basename, const Dim &dim);
void Save(const char *filename, void *data, size_t size);
};
//--------------------------------------------------------------------------
#endif
