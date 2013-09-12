#
# Matrix Multiply example makefile
#
# History: Written  by Tim mattson, August 2010
#          Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Modified by Tom Deakin, July 2013
#

ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99 -ffast-math

LIBS = -lm -lOpenCL -fopenmp

COMMON_DIR = ../../C_common

MMUL_OBJS = matmul.o matrix_lib.o wtime.o err_code.o
EXEC = mult

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
	LIBS = -lm -framework OpenCL -fopenmp
endif

CCFLAGS += -D DEVICE=$(DEVICE)

all: $(EXEC)

mult: $(MMUL_OBJS)
	$(CC) $(MMUL_OBJS) $(LIBS) -o $(EXEC)

wtime.o: $(COMMON_DIR)/wtime.c
	$(CC) -c $^ $(CCFLAGS) -o $@

err_code.o: $(COMMON_DIR)/err_code.c
	$(CC) -c $^ $(CCFLAGS) -o $@

.c.o:
	$(CC) -c $< $(CCFLAGS) -o $@

clean:
	rm -f $(MMUL_OBJS) $(EXEC)
