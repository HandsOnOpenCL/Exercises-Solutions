
CC=gcc

CCFLAGS=-std=c99 -O3

LIBS = -lGL -lglut

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenGL -framework GLUT
endif

displayer: displayer.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f displayer
