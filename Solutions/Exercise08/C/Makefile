
CC=gcc

CCFLAGS=-O3 -lm -std=c99

LIBS = -lOpenCL -fopenmp

COMMON_DIR = ../../C_common

pi_ocl: pi_ocl.c $(COMMON_DIR)/wtime.c $(COMMON_DIR)/err_code.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f pi_ocl
