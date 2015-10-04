#ifndef __DCV_FFTW_H
#define __DCV_FFTW_H
//--------------------------------------------------------------------------
#ifdef USE_DOUBLE
#define	dtype		double
#define	ctype		fftw_complex
#define xfft(x)		fftw_##x
#else
#define	dtype		float
#define	ctype		fftwf_complex
#define xfft(x)		fftwf_##x
#endif
//--------------------------------------------------------------------------
#ifndef dprintf
#define dprintf		printf
#endif
//--------------------------------------------------------------------------
#ifdef DCV_DEBUG
#define dcvout	dprintf
#define dcvput	dputs
#else
#define dcvout(fmt,...)
#define dcvput(txt)
#endif
//-----------------------------------------------------------------------
#ifdef FFTW_NUM_THREADS
#define dcv_init_mthreads()	xfft(init_threads)()
#define dcv_plan_mthreads()	xfft(plan_with_nthreads)(FFTW_NUM_THREADS)
#define dcv_cleanup()		xfft(cleanup_threads)(); xfft(cleanup)()
#else
#define dcv_init_mthreads()
#define dcv_plan_mthreads()
#define dcv_cleanup()
#endif
//--------------------------------------------------------------------------
#define dcv_plan		xfft(plan)
#define dcv_r3d(s,d)	xfft(plan_dft_r2c_3d)(vz,vy,vx,s,d,FFTW_ESTIMATE)
#define dcv_c3d(s,d)	xfft(plan_dft_c2r_3d)(vz,vy,vx,s,d,FFTW_ESTIMATE)
#define dcv_fft(p)		xfft(execute)(fwd_##p)
#define dcv_bft(p)		xfft(execute)(bwd_##p)
#define dcv_close(p)	xfft(destroy_plan)(p)
//--------------------------------------------------------------------------
// dcv fftw memory macroes
//--------------------------------------------------------------------------
#ifdef FFTW_33X
#define dcv_ralloc(n)	xfft(alloc_real)(n)
#define dcv_calloc(n)	xfft(alloc_complex)(n)
#else
#define dcv_ralloc(n)	(dtype *)xfft(malloc)(n * sizeof(dtype))
#define dcv_calloc(n)	(ctype *)xfft(malloc)(n * sizeof(ctype))
#endif
#define dcv_free(b)		xfft(free)(b)
//--------------------------------------------------------------------------
#endif
