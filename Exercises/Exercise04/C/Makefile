
CC=gcc

CCFLAGS=-O3 -lm

LIBS = -lOpenCL -fopenmp

COMMON_DIR = ../../C_common

vadd: vadd_c.c $(COMMON_DIR)/wtime.c $(COMMON_DIR)/err_code.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f vadd
