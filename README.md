## BDKF - 3D Adaptive Spatial deconvolution using Blind Deconvolution with Kalman Filter
----------------------------------------------------------------------------------------

### A. Data

#### 1. Support:

Raw data format with
- 8-bit for source volume
- 32-bit float for DF/PSF volume (distortion function or point-spread function)

#### 2. Volume filename format:

All volume must be named use following format: **volume_name.[width]x[height]x[depth].raw**

Example:  
* source: stack_1_1.512x512x761.raw
* DF/PSF: df.5um.model.64x64x64.raw

### B. Build

#### 1. Requirements:

- Linux OS (tested on Centos 6.x and Fedora 22)
- GNU C Compiler
- GNU Make
- FFTW development library (for CPU code)
- CUDA 7.0 (for GPU code)

#### 2. Compile:

	$ tar xfv bdkf.tar.bz2
	$ cd bdkf
	$ make

3 executable files named **bdkf_cpu**, **bdkf_mft**, and **bdkf_gpu** will be created in
folder **bdkf** after compiling.

#### 3. Compile a specific target

There are 3 targets: **cpu**, **mft**, and **gpu** represent for CPU, CPU with multi-threaded FFTW, and GPU respectively. To compile for specific target, invoke following make command at the prompt

	$ make -C <target>

### C. Execute

In BDKF folder execute following commands

#### 0. Show options

	$ ./bdkf_cpu
	* Usage:
      ./bdkf_cpu [-v] [-c <gpuid>] -i <vsrc> -d <dsrc> -o <vout> -s <vsc> -k <dsc> -n <nit>

         <vsrc>  - source volume (ex: stack_1.512x512x768.raw).
         <dsrc>  - distortion function volume (ex: psf.64x64x64.raw).
         <vout>  - base name of result volume (dimensions & extension will be added automatically).
         <vsc>   - volume stop condition (default: 0.075).
         <dsc>   - distortion function stop condition (default: 0.05).
         <nit>   - number of iterations (default: 100).
	
	$

#### 1. Run BDKF on GPU

	$ ./bdkf_gpu -i /path/to/vol.raw -d /path/to/df.raw -o result -s 0.075 -k 0.05 -n 100

#### 2. Run BDKF on CPU

	$ ./bdkf_cpu -i /path/to/vol.raw -d /path/to/df.raw -o result -s 0.075 -k 0.05 -n 100

#### 3. Run BDKF on CPU with multi-threaded FFTW

	$ ./bdkf_mft -i /path/to/vol.raw -d /path/to/df.raw -o result -s 0.075 -k 0.05 -n 100

#### 4. Example

Suppose all **stack & DF files** are stored in **bdkf** folder then

	$ ./bdkf_gpu -i stack.512x512x761.raw -d df.64x64x64.raw -o s11df.gpu -s 0.075 -k 0.05 -n 100

The output will be **s11df.gpu.512x512x761.raw**, dimension and extension will be added to result volume name automatically.

Or simply use **run.sh** shell-script to invoke **cpu / mft / gpu** executables with default parameters:

	$ ./run.sh cpu stack_1_1.512x512x761.raw df.64x64x64.raw s11df.cpu.result
	$ ./run.sh mft stack_1_1.512x512x761.raw df.64x64x64.raw s11df.cpu.result
	$ ./run.sh gpu stack_1_1.512x512x761.raw df.64x64x64.raw s11df.gpu.result

### D. Testing with sample data

#### 0. Get sample data

	$ wget https://www.dropbox.com/s/oxeolgvoqii7jjt/stack_1_1.512x512x761.raw
	$ wget https://www.dropbox.com/s/sag5xke3un8l0ba/df.64x64x64.raw

#### 1. Test GPU code on GTX 970

#### 2. Test CPU code CPU Intel Core i7 4.0GHz

	$ ./bdkf_gpu -i stack.512x512x761.raw -d df.64x64x64.raw -o s11df.gpu -s 0.075 -k 0.05 -n 100

#### 3. Test CPU code with multi-threaded FFTW on Intel Core i7 Q720M 1.6GHz (Sony Laptop)

	$ ./bdkf_mft -i stack.512x512x761.raw -d df.64x64x64.raw -o mft -s 0.075 -k 0.05 -n 100
	* Options:
	  - DF/PSF: ../samples/df.64x64x64.raw
	  - Volume: ../samples/stack_1_1.512x512x761.raw
	  - Output basename: mft
	  - Stop conditions: 7.5E-02 5.0E-02
	  - Number of iterations: 100
	  - Verbose: no
	
	- Offset: [32, 32, 32]
	- Block size: 64x64x64
	- Working size: 128x128x128
	   < 16384 voxels/plane, 2097152 voxels/volume >

	<...................... some verbose outputs ......................>

	,--[ Done ]--------
	| Saving... done.
	`-- Time: 1.08s ---
	Total time: 252.659s
