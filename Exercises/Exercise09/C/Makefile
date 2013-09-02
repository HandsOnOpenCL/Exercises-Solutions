CC=gcc

CCFLAGS=-lm -O3

LIBS = -fopenmp

COMMON_DIR = ../../C_common

pi: pi.c $(COMMON_DIR)/wtime.c
	$(CC) $^ $(CCFLAGS) -o $@

clean:
	rm -f pi
