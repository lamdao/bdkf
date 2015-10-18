#---------------------------------------------------------------------------
# Makefile for GPU code
#---------------------------------------------------------------------------
# Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
#---------------------------------------------------------------------------
SRCS=..
INCS=-I. -I$(SRCS)
CORE=bdkf_gpu.a
CUDA=$(CORE) -lcudart -lcufft
LIBS=$(CUDA) -lstdc++
DEFS=-DNUM_THREADS=1024
ARCH=-m64

ifeq "$(USE_M32)" "1"
ARCH=-m32
endif

# if compile with make DEBUG=1
ifeq "$(DEBUG)" "1"
OPTS=$(ARCH) -O0 -g3
else # otherwise use optimize flags
OPTS=$(ARCH) -march=native -mtune=native -O3 -s -fomit-frame-pointer
endif

TARGET=$(SRCS)/bdkf_gpu
COBJ=timer.o options.o rawfile.o bdkf_main.o
GOBJ=bdkf_core.o kuda_utils.o

ifeq "$(SM)" ""
SM=20
endif
# GPU Architecture & Compute capability flags
GARCH=-arch compute_$(SM) -code=sm_$(SM),compute_$(SM)

# nVidia compiller flags
NVCF=$(DEFS) $(GARCH) $(INCS)

%.o: $(SRCS)/%.cpp
	$(CXX) $(DEFS) $(OPTS) $(INCS) -c $<

%.o: %.cu
	nvcc $(NVCF) -c $<

all: $(TARGET)

clean:
	rm -f *.o

distclean: clean
	rm -f *.a
	rm -f $(TARGET)

$(TARGET): $(COBJ) $(CORE)
	nvcc -O3 $(ARCH) -Xlinker -s -o $(TARGET) $(COBJ) $(LIBS)

kuda_utils.o: kuda_utils.cu kuda_utils.h
bdkf_core.o: bdkf_core.cu $(SRCS)/bdkf_core.h
bdkf_main.o: $(SRCS)/bdkf_main.cpp $(SRCS)/volume.h $(SRCS)/typedefs.h
rawfile.o: $(SRCS)/rawfile.cpp $(SRCS)/rawfile.h
options.o: $(SRCS)/options.cpp $(SRCS)/options.h
timer.o: $(SRCS)/timer.cpp $(SRCS)/timer.h

$(CORE): $(GOBJ)
	@rm -f $(CORE)
	@ar crs $(CORE) $(GOBJ)

