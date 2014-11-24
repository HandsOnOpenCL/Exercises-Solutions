
ifndef CC
	CC = gcc
endif

CCFLAGS=-std=c99

LIBS = -lOpenCL

COMMON_DIR = ../../C_common

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

DeviceInfo: DeviceInfo.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@


clean:
	rm -f DeviceInfo
