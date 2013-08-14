
CC=gcc

CCFLAGS=-std=c99 -g -I /usr/local/cuda/include

LIBS = -lOpenCL

COMMON_DIR = ../../C_common

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CCFLAGS += -DAPPLE
	LIBS = -framework OpenCL
endif

DeviceInfo: DeviceInfo.c $(COMMON_DIR)/err_code.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f DeviceInfo
