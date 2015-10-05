//--------------------------------------------------------------------------
// options.cpp - command line parameters parsing utilities
// 
// This file provide simple functions to parse provided parameters from
// command line. It also defines some common variables that are used in
// the main processing file bdkf_main.cpp
//--------------------------------------------------------------------------
// Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
//--------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//--------------------------------------------------------------------------
#include "options.h"
//--------------------------------------------------------------------------
bool verbose = false;
//--------------------------------------------------------------------------
int niters = 100;
int bsize = 64;
//--------------------------------------------------------------------------
dtype vstop = (dtype)0.075;
dtype kstop = (dtype)0.05;
//--------------------------------------------------------------------------
int devNo = 0;
//--------------------------------------------------------------------------
const char *vsource = NULL;			// source volume filename
const char *dsource = "psf.miv";	// DF/PSF volume filename
const char *voutput = "dcv.miv";	// result volume filename
//--------------------------------------------------------------------------
void Usage(char *prog)
{
	#define pad	"         " 
	printf("* Usage:\n"
		"  %s [-v] [-c <gpuid>] -i <vsrc> -d <dsrc> -o <vout> -s <vsc> -k <dsc> -n <nit>\n\n"
		pad	"<vsrc>  - source volume (ex: stack_1.512x512x768.raw).\n"
		pad	"<dsrc>  - distortion function volume (ex: psf.64x64x64.raw).\n"
		pad	"<vout>  - base name of result volume (dims & ext will be added).\n"
		pad	"<vsc>   - volume stop condition (default: 0.075).\n"
		pad	"<dsc>   - distortion function stop condition (default: 0.05).\n"
		pad	"<nit>   - number of iterations (default: 100).\n\n", prog);
	exit(0);
}
//--------------------------------------------------------------------------
void GetOptions(int n, char **params)
{
	int opt;
	while ((opt = getopt(n, params, "vc:i:o:d:s:k:n:b:")) != -1)
	switch (opt) {
		case 'v':
			verbose = true;
			break;
		case 'c':
			devNo = atoi(optarg);
			if (devNo < 0) devNo = 0;
			break;
		case 'i':
			vsource = optarg;
			break;
		case 'o':
			voutput = optarg;
			break;
		case 'd':
			dsource = optarg;
			break;
		case 's':	// volume stop condition
			vstop = (dtype)atof(optarg);
			break;
		case 'k':	// df stop condition
			kstop = (dtype)atof(optarg);
			break;
		case 'n':	// number of iterations
			niters = atoi(optarg);
			break;
		case 'b':
			bsize = atoi(optarg);
			if (bsize < 0 || bsize > 128)
				bsize = 64;
			break;
		default:
			Usage(params[0]);
	}

	if (!vsource) {
		Usage(params[0]);
	}

	printf(
		"* Options:\n"
		"  - DF/PSF: %s\n"
		"  - Volume: %s\n"
		"  - Output basename: %s\n"
		"  - Stop conditions: %1.1E %1.1E\n"
		"  - Number of iterations: %d\n"
		"  - Verbose: %s\n\n",
		dsource, vsource, voutput,
		vstop, kstop, niters,
		verbose ? "yes" : "no");
}
//--------------------------------------------------------------------------
