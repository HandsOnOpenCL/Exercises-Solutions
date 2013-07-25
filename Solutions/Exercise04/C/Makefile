
CC=gcc

CCFLAGS=-O3 -lm

LIBS = -lOpenCL

COMMON_DIR = ../../C_common

vadd_chain: vadd_chain.c $(COMMON_DIR)/err_code.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f vadd
