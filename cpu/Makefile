#------------------------------------------------------------------------
# Makefile for CPU code
#------------------------------------------------------------------------
# Author: Lam H. Dao <daohailam(at)yahoo(dot)com>
#------------------------------------------------------------------------
STD=-std=c++11

SRCS=..
INCS=-I. -I$(SRCS)
CORE=bdkf_cpu.a
LIBS=$(CORE) -lfftw3f -lstdc++
ARCH=-m64

ifeq "$(USE_M32)" "1"
ARCH=-m32
endif

ifeq "$(DEBUG)" "1"
# Debug flags
OPTS=$(ARCH) -O0 -g3 $(STD)
else
# Optimize flags
OPTS=$(ARCH) -march=native -mtune=native -O3 -s -fomit-frame-pointer $(STD)
endif

TARGET=$(SRCS)/bdkf_cpu
COBJ=timer.o options.o rawfile.o bdkf_main.o

%.o: $(SRCS)/%.cpp
	$(CXX) $(OPTS) $(INCS) -c $<

%.o: %.cpp
	$(CXX) $(OPTS) $(INCS) -c $<

all: $(TARGET)

clean:
	rm -f *.o

distclean: clean
	rm -f *.a
	rm -f $(TARGET)

$(TARGET): $(COBJ) $(CORE)
	$(CXX) $(OPTS) -o $(TARGET) $(COBJ) $(LIBS)

bdkf_core.o: bdkf_core.cpp $(SRCS)/bdkf_core.h DcvCenterShift.h DcvConvolver.h DcvFFTW.h DcvUpdater.h Resources.h 
bdkf_main.o: $(SRCS)/bdkf_main.cpp $(SRCS)/volume.h $(SRCS)/typedefs.h
rawfile.o: $(SRCS)/rawfile.cpp $(SRCS)/rawfile.h
options.o: $(SRCS)/options.cpp $(SRCS)/options.h
timer.o: $(SRCS)/timer.cpp $(SRCS)/timer.h

$(CORE): bdkf_core.o Resources.o
	@rm -f $(CORE)
	@ar crs $(CORE) bdkf_core.o Resources.o

