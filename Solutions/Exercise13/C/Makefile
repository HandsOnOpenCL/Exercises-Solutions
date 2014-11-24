
ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99

LIBS = -lOpenCL -lm

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

gameoflife: gameoflife.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@


clean:
	rm -f gameoflife *.o
