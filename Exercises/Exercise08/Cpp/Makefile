CC=g++

CCFLAGS=-O3 -std=c++11 -std=gnu++11

LIBS = -lm

CPP_COMMON = ../../Cpp_common

INC = -I $(CPP_COMMON)

pi: pi.cpp
	$(CC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f pi
