
CC=gcc

CCFLAGS=-O3 -lm -std=c99

LIBS = -lOpenCL -fopenmp

COMMON_DIR = ../../C_common

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

ifdef APPLE
	CCFLAGS += -DAPPLE
	LIBS = -framework OpenCL -fopenmp
endif

CCFLAGS += -D DEVICE=$(DEVICE)

pi_ocl: pi_ocl.c $(COMMON_DIR)/wtime.c $(COMMON_DIR)/err_code.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f pi_ocl
