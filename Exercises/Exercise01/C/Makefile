
CC=gcc

CCFLAGS=-std=c99 -g -I /usr/local/cuda/include

LIBS = -lOpenCL

COMMON_DIR = ../../C_common

ifdef APPLE
	CCFLAGS += -DAPPLE
	LIBS = -framework OpenCL
endif

DeviceInfo: DeviceInfo.c $(COMMON_DIR)/err_code.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f DeviceInfo
