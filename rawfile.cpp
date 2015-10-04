//--------------------------------------------------------------------------
// rawfile.cpp - a 3D volume in raw-format (pure data only) controller
// 
// This file provides simple functions for handling 3D volume in raw-format
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
//--------------------------------------------------------------------------
#include <memory.h>
#include "point.h"
//--------------------------------------------------------------------------
using namespace std;
//--------------------------------------------------------------------------
namespace RawFile {
//--------------------------------------------------------------------------
// Report filename format error and exit
//--------------------------------------------------------------------------
void DimError(const char *filename)
{
	cout << "* Error:" << '\n';
	cout << "  Cannot determine volume size for " << filename << '\n';
	cout << "  Volume file name must have following format:\n";
	cout << "      Volname.WxHxD.raw -- W, H, D are 3 dimensions of Volname.\n";
	cout << "    Ex:\n";
	cout << "      Stack_1_1.512x512x768.raw\n\n";
	exit(0);
}
//--------------------------------------------------------------------------
// Extract volume dimensions from filename
// Filename format: name.<width>x<height>x<depth>.raw
//   Ex: stack_1.512x512x768.raw
//--------------------------------------------------------------------------
Dim DimExtract(const char *filename)
{
	string fn(filename);
	char *ns = (char *)fn.c_str();
	char *p = strrchr(ns, '.');			// skip extension
	if (!p) DimError(filename);			// if no ext => error
	*p = 0; p = strrchr(ns, '.');		// find dim-start
	if (!p) DimError(filename);			// if not found => error

	char *ws = strtok(p+1, "x");
	char *hs = strtok(0, "x");
	char *ds = strtok(0, ".");
	if (!ws || !hs || !ds) {
		DimError(filename);
	}

	int w = atoi(ws);
	int h = atoi(hs);
	int d = atoi(ds);

	if (w <= 0 || h <= 0 || d <= 0) {
		DimError(filename);
	}
	return Dim(w, h, d);
}
//--------------------------------------------------------------------------
void Load(const char *filename, void *data, size_t size)
{
	ifstream fp(filename, ios::binary);
	if (fp) fp.read((char *)data, size);
}
//--------------------------------------------------------------------------
char *FormatName(const char *basename, const Dim &dim)
{
	static char fn[256];
	snprintf(fn, sizeof(fn), "%s.%dx%dx%d.raw",
				basename, dim.x, dim.y, dim.z);
	return fn;
}
//--------------------------------------------------------------------------
void Save(const char *filename, void *data, size_t size)
{
	ofstream fp(filename, ios::binary);
	if (fp) fp.write((char *)data, size);
}
//--------------------------------------------------------------------------
} // namespace RawFile
