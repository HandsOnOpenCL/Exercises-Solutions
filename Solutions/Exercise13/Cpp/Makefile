
ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = ../../Cpp_common

CCFLAGS=-std=c++11 -O3

INC = -I $(CPP_COMMON)

LIBS = -lOpenCL -lrt

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

all: gameoflife

gameoflife: gameoflife.cpp err_code.o
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

err_code.o: $(CPP_COMMON)/err_code.c
	$(CPPC) -c $^ $(CCFLAGS) -o $@

clean:
	rm -f gameoflife *.o
