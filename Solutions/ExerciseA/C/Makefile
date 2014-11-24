
ifndef CC
	CC = gcc
endif

CCFLAGS=-g -std=c99

LIBS = -lOpenCL -fopenmp -lm

COMMON_DIR = ../../C_common

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
	LIBS = -framework OpenCL -lm
endif

CCFLAGS += -D DEVICE=$(DEVICE)

pi_vocl: pi_vocl.c $(COMMON_DIR)/wtime.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@


clean:
	rm -f pi_vocl
