//--------------------------------------------------------------------------
// volume.h - Volume/Stack image manipulation class
//
// The file provides a class for handling image volume. The whole project
// depends on this class. It provides a simple way to load/save (using
// rawfile.{h,cpp}), expand, pad, store sub volume and type conversion.
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#ifndef __VOLUME_H
#define __VOLUME_H
//--------------------------------------------------------------------------
#include <memory.h>
#include <cassert>
//--------------------------------------------------------------------------
#include "point.h"
#include "rawfile.h"
//--------------------------------------------------------------------------
template<class VT>
class Volume {
public:
	Dim dim;
private:
	VT *data;
	struct {
		size_t slice;
		size_t total;
		size_t bytes;
	} vsize;

	void Create(void *buffer) {
		vsize.slice = (size_t)dim.x * dim.y;
		vsize.total = vsize.slice * dim.z;
		vsize.bytes = vsize.total * sizeof(VT);
		if (buffer)
			data = (VT *)buffer;
		else {
			data = new VT[vsize.total];
			assert(data != 0);
		}
	}

	void Create(int w, int h, int d, void *buffer = 0) {
		dim = Dim(w, h, d);
		Create(buffer);
	}

	void Create(const Dim &d, void *buffer = 0) {
		dim = d;
		Create(buffer);
	}
public:
	Volume(): dim(), data(0) { vsize.slice = vsize.total = vsize.bytes = 0; }
	Volume(int w, int h, int d, void *vol = 0) { Create(w, h, d, vol); }
	Volume(const Dim &dim, void *vol = 0) { Create(dim, vol); }
	Volume(const char *filename) {
		Create(RawFile::DimExtract(filename));
		RawFile::Load(filename, data, Size());
	}
	~Volume() {
		if (data) delete [] data;
	}

	void Recreate(const Dim &dim, void *vol = 0) {
		if (this->dim == dim && vol == 0)
			return;
		if (data) delete [] data;
		Create(dim, vol);
	}

	size_t Size() { return vsize.bytes; }
	void Clear() { memset(data, 0, Size()); }
	void Detach() { data = NULL; }

	void Save(const char *basename) {
		char *fn = RawFile::FormatName(basename, dim);
		RawFile::Save(fn, data, Size());
	}

	operator void* () { return data; }
	operator VT* () { return data; }
	VT& operator () (int x, int y, int z) {
		return data[GetIndex(x, y, z)];
	}

	size_t GetIndex(int x, int y, int z) {
		return (size_t)z * vsize.slice + (size_t)y * dim.x + x;
	}

	Volume<VT> GetPadVolume(int w, int h, int d) {
		size_t p = (size_t)w * h;
		size_t s = p * d;
		VT *rv = new VT[s]; assert(rv != NULL);
		memset(rv, 0, s * sizeof(VT));

		Dim ofs = (Dim(w, h, d) - dim) / 2;
		size_t bs = (size_t)ofs.z * p + (size_t)ofs.y * w + ofs.x;
		for (int z = 0; z < dim.z; z++) {
			size_t zn = z * vsize.slice;
			size_t zr = z * p;
			for (int y = 0; y < dim.y; y++) {
				size_t yn = y * dim.x;
				size_t yr = y * w;
				for (int x = 0; x < dim.x; x++) {
					size_t idx = zn + yn + x;
					size_t odx = zr + yr + x;
					rv[bs + odx] = data[idx];
				}
			}
		}
		return Volume<VT>(w, h, d, rv);
	}

	Volume<VT> GetPadVolume(const Dim &dim) {
		return GetPadVolume(dim.x, dim.y, dim.z);
	}

	Volume<VT> Expand(int w, int h, int d) {
		Volume<VT> vol(w, h, d);
		vol.Clear();
		vol.Store(0, 0, 0, *this);

		VT *v = vol;
		size_t bdx = vol.GetIndex(0, 0, dim.z - 1);
		size_t ddx = bdx + vol.vsize.slice;
		for (int n = dim.z; n < d; n++) {
			memcpy(&v[ddx], &v[bdx], vol.vsize.slice * sizeof(VT));
			ddx += vol.vsize.slice;
		}
		return vol;
	}

	Volume<VT> Expand(const Dim &dim) {
		return Expand(dim.x, dim.y, dim.z);
	}

	Volume<uchar> ByteScale() {
		VT vmax = data[0], vmin = data[0];
		for (size_t n = 1; n < vsize.total; n++) {
			VT v = data[n];
			if (vmax < v) vmax = v;
			if (vmin > v) vmin = v;
		}
		vmax -= vmin;

		uchar *rv = new uchar[vsize.total]; assert(rv != NULL);
		for (size_t n = 0; n < vsize.total; n++) {
			rv[n] = (uchar)(255 * (data[n] - vmin) / vmax);
		}
		return Volume<uchar>(dim, rv);
	}

	void Store(int x, int y, int z, const Volume<VT> &vol) {
		VT *src = vol.data;
		size_t ofs = GetIndex(x, y, z);
		for (int z = 0; z < vol.dim.z; z++) {
			VT *slice = &data[ofs];
			for (int y = 0; y < vol.dim.y; y++) {
				memcpy(slice, src, sizeof(VT) * vol.dim.x);
				src += vol.dim.x;
				slice += dim.x;
			}
			ofs += vsize.slice;
		}
	}

	void Store(const Point<int>& pt, const Volume<VT> &vol) {
		Store(pt.x, pt.y, pt.z, vol);
	}
};
//--------------------------------------------------------------------------
#endif
