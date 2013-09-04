
CC = nvcc

CCFLAGS = -O3

LIBS =

vadd: vadd.cu
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f vadd
