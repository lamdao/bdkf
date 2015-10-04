//--------------------------------------------------------------------------
// Resources.cpp - Precalculate volume index to map position to index back
//                 and forth for speedup
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include <stdio.h>
#include <memory.h>
//--------------------------------------------------------------------------
#include "Resources.h"
//--------------------------------------------------------------------------
using namespace std;
//--------------------------------------------------------------------------
vector<int> Resources::xts;
vector<int> Resources::yts;
vector<int> Resources::zts;
vector<bool> Resources::bmp;
//--------------------------------------------------------------------------
int Resources::mx = 0;
int Resources::my = 0;
int Resources::mz = 0;
//--------------------------------------------------------------------------
int Resources::vx = 0;
int Resources::vy = 0;
int Resources::vz = 0;
//--------------------------------------------------------------------------
int Resources::cx = 0;
int Resources::cy = 0;
int Resources::cz = 0;
//--------------------------------------------------------------------------
size_t Resources::vp = 0;
size_t Resources::vn = 0;
size_t Resources::vc = 0;
//--------------------------------------------------------------------------
double Resources::noisevar = 0.05;
double Resources::gain = 0.5;
//--------------------------------------------------------------------------
template<class T>
void Resize(T &v, size_t n)
{
	T empty(n);
	v.swap(empty);
}
//--------------------------------------------------------------------------
void Resources::Init()
{
	Resize(xts, vn);
	Resize(yts, vn);
	Resize(zts, vn);
	Resize(bmp, vn);

	mx = vx - 2;
	my = vy - 2;
	mz = vz - 2;

	FillMap();
}
//--------------------------------------------------------------------------
void Resources::FillMap()
{
	for (size_t n = 0; n < vn; n++) {
		int x, y, z;
		size_t nt = n;
		zts[n] = z = (int)(nt / vp); nt = nt % vp;
		yts[n] = y = (int)(nt / vx);
		xts[n] = x = (int)(nt % vx);
		bmp[n] = x > 0 && x <= mx && y > 0 && y <= my && z > 0 && z <= mz;
	}
}
//--------------------------------------------------------------------------
