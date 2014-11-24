
ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99

LIBS = -lOpenCL -fopenmp -lm

COMMON_DIR = ../../C_common

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL -lm
endif


pi_ocl: pi_ocl.c $(COMMON_DIR)/wtime.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@


clean:
	rm -f pi_ocl
