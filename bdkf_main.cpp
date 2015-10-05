//--------------------------------------------------------------------------
// bdkf_main.cpp - main file of both CPU & GPU blind deconvolution
//
// In this file, an adaptive spatial blind deconvolution algorithm is
// implemented. The algorithm is represented in the for-loops of main(),
// in which a whole image volume is divided into multiple sub-blocks and
// deconvoled using the approach introduced in this paper:
//
// http://onlinelibrary.wiley.com/doi/10.1111/jmi.12281/abstract
// 
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
//--------------------------------------------------------------------------
#include "bdkf_core.h"
#include "options.h"
#include "volume.h"
#include "timer.h"
//--------------------------------------------------------------------------
// * Note:
//   vsource, dsource, voutput, devNo are declared in files options.{h,cpp}
//   These variables will be initialized in GetOptions function
//   (See options.cpp)
//--------------------------------------------------------------------------
void Init(int n, char **params)
{
	// Parsing parameters from command line
	GetOptions(n, params);

	bdkf::Device::Init();
}
//--------------------------------------------------------------------------
void SaveResult(const Dim &dim)
{
	timer::Start();
	printf(" \n,--[ Done ]--------\n| Saving... "); fflush(stdout);

	// Get back result buffer from computing memory to CPU memory
	dtype *result = bdkf::GetFinalResult();
	// Create volume from result buffer
	Volume<dtype> rv(dim, result);
	// Convert to 8-bit and save
	rv.ByteScale().Save(voutput);
	rv.Detach();

	printf("done.\n");
	printf("`-- Time: %.2fs ---\n", timer::Check());
}
//--------------------------------------------------------------------------
void Done()
{
	// Release all allocated memory buffers
	bdkf::Done();

	printf("Total time: %.3fs\n", timer::GetRuntime());
}
//--------------------------------------------------------------------------
int main(int argc, char **argv)
{
	Init(argc, argv);

	Volume<uchar> src(vsource);
	Volume<dtype> psf(dsource);			// dimensions of DF/PSF must be 2^n

	// Calculating buffer sizes and offsets
	Dim odm = src.dim;						// original volume size
	Dim cbs = Dim(bsize,bsize,bsize);		// calculating block size
	Dim rup = (odm/cbs+((odm%cbs)>0))*cbs;	// round-up padding size
	Dim wbs = (2*cbs)>>psf.dim;				// working block size:
											// (a.k.a deconvolution size)
											//    = max(2 * cbs, psf.dim)
	Dim &vdim = rup;						// dimensions of working volume
	Pos ofs = cbs/2;						// start offset after padding

	ofs.Show("- Offset");

	// Initialize all memory buffers in Processor (GPU/CPU)
	// that are needed for deconvolution of
	// a volume with dimensions = wbs
	bdkf::Init(wbs, cbs);

	// Put Source volume to Processor memory
	Volume<uchar> vol = src.Expand(vdim);
	bdkf::InitSourceBuffer(vol, vdim);

	// Put DF0 to Processor memory
	Volume<dtype> df0 = psf.GetPadVolume(wbs);
	bdkf::InitDF0(df0);

	// Create result buffer in Processor memory
	bdkf::CreateResultBuffer();

	const char *spinner = "|\b/\b-\b\\\b";
	int nc = 0;

	printf("\nZ = ");
	// This for-loops block represents the
	// adaptive spatial blind deconvolution
	// strategy (block-by-block deconvolution)
	for (int z = ofs.z; z < vdim.z; z += cbs.z) {
		printf("%d ", z); fflush(stdout);
		for (int y = ofs.y; y < vdim.y; y += cbs.y) {
			for (int x = ofs.x; x < vdim.x; x += cbs.x) {
				Pos cp(x, y, z);
				Pos bp = cp - ofs;
				// Crop block at <bp - ofs> from source volume
				// to working buffer on Processor memory
				bdkf::CropSourceBlock(bp - ofs);

				// Reset timer counter
				if (verbose) {
					timer::Start();
				}

				// Execute blind deconvolution on current block
				dtype rvstop, rkstop;
				int ni = bdkf::Exec(rvstop, rkstop);

				// Check elapsed time and report
				// [#blk_no: elapsed_time - n_iter, v_residual, k_residual]
				if (verbose) {
					printf("[#%04d: %.1fs - %03d,%.2f,%.2f] ",
							++nc,timer::Check(),ni,100*rvstop,100*rkstop);
					if (nc % 4 == 0) printf("\n    ");
				} else {
					write(1, &spinner[(nc % 4) << 1], 2);
					++nc;
				}
				// Store deconvolution result from working buffer
				// to result buffer at <bp> directly on GPU memory
				bdkf::SaveResultBlock(bp);
			}
		}
	}

	SaveResult(odm);
	Done();

	return 0;
}
