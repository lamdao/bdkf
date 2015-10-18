#------------------------------------------------------------------------
# Makefile for Multi-threaded FFTW code
#------------------------------------------------------------------------
# Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
#------------------------------------------------------------------------
STD=-std=c++11

SRCS=..
INCS=-I. -I$(SRCS)
CORE=bdkf_mft.a
LIBS=$(CORE) -lfftw3f -lfftw3f_threads -lstdc++ -lpthread
DEFS=-DFFTW_NUM_THREADS=$(shell grep -c processor /proc/cpuinfo)
ARCH=-m64

ifeq "$(USE_M32)" "1"
ARCH=-m32
endif

# if compile with make DEBUG=1
ifeq "$(DEBUG)" "1"
OPTS=$(ARCH) -O0 -g3 $(STD)
else # otherwise use optimize flags
OPTS=$(ARCH) -march=native -mtune=native -O3 -s -fomit-frame-pointer $(STD)
endif

TARGET=$(SRCS)/bdkf_mft
COBJ=timer.o options.o rawfile.o bdkf_main.o

%.o: $(SRCS)/%.cpp
	$(CXX) $(OPTS) $(INCS) -c $<

%.o: %.cpp
	$(CXX) $(DEFS) $(OPTS) $(INCS) -c $<

all: $(TARGET)

clean:
	rm -f *.o

distclean: clean
	rm -f *.a
	rm -f $(TARGET)

$(TARGET): $(COBJ) $(CORE)
	$(CXX) $(OPTS) -o $(TARGET) $(COBJ) $(LIBS)

bdkf_core.o: bdkf_core.cpp $(SRCS)/bdkf_core.h
bdkf_main.o: $(SRCS)/bdkf_main.cpp $(SRCS)/volume.h $(SRCS)/typedefs.h
rawfile.o: $(SRCS)/rawfile.cpp $(SRCS)/rawfile.h
options.o: $(SRCS)/options.cpp $(SRCS)/options.h
timer.o: $(SRCS)/timer.cpp $(SRCS)/timer.h

$(CORE): bdkf_core.o Resources.o
	@rm -f $(CORE)
	@ar crs $(CORE) bdkf_core.o Resources.o

