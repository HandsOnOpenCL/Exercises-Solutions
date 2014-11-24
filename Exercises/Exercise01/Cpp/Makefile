
ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = ../../Cpp_common

CCFLAGS=

INC = -I $(CPP_COMMON)

LIBS = -lOpenCL

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	LIBS = -framework OpenCL
endif

DeviceInfo: DeviceInfo.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f DeviceInfo
