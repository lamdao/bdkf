//--------------------------------------------------------------------------
// point.h - a simple class for 3D point manipulation
// 
// This file provides a simple class for handling 3D point that is used in
// implementation of block-by-block deconvolution algorithm.
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __VOL_POINT_h
#define __VOL_POINT_h
//--------------------------------------------------------------------------
#include <stdio.h>
#include <typeinfo>
//--------------------------------------------------------------------------
#define Range	Point
#define Vector	Point
//--------------------------------------------------------------------------
#define ptmax(a, b)	((a) > (b) ? (a) : (b))
#define ptmin(a, b)	((a) < (b) ? (a) : (b))
//--------------------------------------------------------------------------
template<class T>
class Point
{
public:
	T x, y, z;

	Point(): x(0), y(0), z(0) {}
	Point(T xx, T yy, T zz): x(xx), y(yy), z(zz) {}
	Point(const Point& p): x(p.x), y(p.y), z(p.z) {}

	Point& operator = (const Point& p) {
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}
#ifdef __CUDACC__
	operator int3 () const {
		return make_int3(x, y, z);
	}
#endif
	Point operator + (const Point& p) const {
		return Point(x + p.x, y + p.y, z + p.z);
	}

	Point operator + (const T n) const {
		return Point(x + n, y + n, z + n);
	}

	Point operator - (const Point& p) const {
		return Point(x - p.x, y - p.y, z - p.z);
	}

	Point operator - (const T n) const {
		return Point(x - n, y - n, z - n);
	}

	Point operator * (const Point& p) const {
		return Point(x * p.x, y * p.y, z * p.z);
	}

	Point operator * (const T n) const {
		return Point(x * n, y * n, z * n);
	}

	Point operator / (const Point& p) const {
		return Point(x / p.x, y / p.y, z / p.z);
	}

	Point operator / (const T n) const {
		return Point(x / n, y / n, z / n);
	}

	Point operator % (const Point& p) const {
		return Point(x % p.x, y % p.y, z % p.z);
	}

	Point operator % (const T n) const {
		return Point(x % n, y % n, z % n);
	}

	bool operator == (const Point& p) const {
		return x == p.x && y == p.y && z == p.z;
	}

	bool operator == (const T n) const {
		return x == n && y == n && z == n;
	}

	Point operator > (const Point& p) const {
		return Point(x > p.x, y > p.y, z > p.z);
	}

	Point operator > (const T n) const {
		return Point(x > n, y > n, z > n);
	}

	Point operator < (const Point& p) const {
		return Point(x < p.x, y < p.y, z < p.z);
	}

	Point operator < (const T n) const {
		return Point(x < n, y < n, z < n);
	}

	// Closure Max
	Point operator >> (const Point& p) const {
		return Point(ptmax(x, p.x), ptmax(y, p.y), ptmax(z, p.z));
	}

	Point operator >> (const T n) const {
		return Point(ptmax(x, n), ptmax(y, n), ptmax(z, n));
	}

	// Inner Min
	Point operator << (const Point& p) const {
		return Point(min(x, p.x), min(y, p.y), min(z, p.z));
	}

	Point operator << (const T n) const {
		return Point(min(x, n), min(y, n), min(z, n));
	}

	// Friend operators
	friend Point operator + (const T n, const Point& p) {
		return Point(n + p.x, n + p.y, n + p.z);
	}

	friend Point operator - (const T n, const Point& p) {
		return Point(n - p.x, n - p.y, n - p.z);
	}

	friend Point operator * (const T n, const Point& p) {
		return Point(n * p.x, n * p.y, n * p.z);
	}

	friend Point operator / (const T n, const Point& p) {
		return Point(n / p.x, n / p.y, n / p.z);
	}

	// Utilities
	double Length() const {
		return sqrt(x*x + y*y + z*z);
	}

	Point<double> Normalize() const {
		double l = Length();
		if (l == 0) return Point<double>(0,0,0);
		return Point<double>(x/l, y/l, z/l);
	}

	void Show(const char *name = NULL) const {
		printf("%s: [%d, %d, %d]\n",
				name != NULL ? name : typeid(this).name(), x, y, z);
	}
};
//--------------------------------------------------------------------------
typedef Point<int> Dim;
typedef Point<int> Pos;
//--------------------------------------------------------------------------
inline Point<int> point(int x, int y, int z)
{
	return Point<int>(x, y, z);
}
//--------------------------------------------------------------------------
#endif
