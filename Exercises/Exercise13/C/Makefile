
ifndef CC
	CC = gcc
endif

CCFLAGS = -O3 -std=c99

LIBS = -fopenmp

gameoflife: gameoflife.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f gameoflife
