
ifndef CC
	CC = gcc
endif

CCFLAGS = -O3 -std=c99

LIBS = -fopenmp

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
        LIBS = 
endif

gameoflife: gameoflife.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f gameoflife
