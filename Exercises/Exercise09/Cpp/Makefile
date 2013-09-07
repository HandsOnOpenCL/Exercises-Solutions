
ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-O3 -std=c++11

LIBS = -lm -lrt

CPP_COMMON = ../../Cpp_common

INC = -I $(CPP_COMMON)

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -DAPPLE -stdlib=libc++
endif

pi: pi.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f pi
