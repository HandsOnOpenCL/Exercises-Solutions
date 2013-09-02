
CC = nvcc

CCFLAGS = -O3

LIBS =

gameoflife: gameoflife.cu
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f gameoflife
