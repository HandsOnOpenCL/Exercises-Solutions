
CC = gcc

#CCFLAGS = -O3
CCFLAGS = -g -std=c99

LIBS = -lm

gameoflife: gameoflife.c
	$(CC) $^ $(CCFLAGS) -o $@

clean:
	rm -f gameoflife
