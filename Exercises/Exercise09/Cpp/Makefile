
ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-O3 -std=c++11

LIBS = -lm -lrt

CPP_COMMON = ../../Cpp_common

INC = -I $(CPP_COMMON)

pi: pi.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f pi
