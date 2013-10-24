
ifndef CC
	CC = gcc
endif

CCFLAGS=-lm -O3

LIBS = -fopenmp

COMMON_DIR = ../../C_common

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
       LIBS =  
endif

pi: pi.c $(COMMON_DIR)/wtime.c
	$(CC) $^ $(CCFLAGS) -o $@

clean:
	rm -f pi
