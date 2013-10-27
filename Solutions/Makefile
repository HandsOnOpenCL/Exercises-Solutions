
# This makefile will produce all the C binaries
# in their respective directories

CEXES = Exercise04/C/vadd_chain Exercise05/C/vadd_abc \
		Exercise06/C/mult Exercise07/C/mult \
		Exercise08/C/mult Exercise09/C/pi_ocl \
		Exercise13/C/gameoflife ExerciseA/C/pi_vocl

CPPEXES = Exercise04/Cpp/vadd_chain Exercise05/Cpp/vadd_abc \
		Exercise06/Cpp/mult Exercise07/Cpp/mult \
		Exercise08/Cpp/mult Exercise08/Cpp/pi_ocl \
		Exercise13/Cpp/gameoflife ExerciseA/Cpp/pi_vocl

# Change this variable to specify the device type in all
# the Makefile to the OpenCL device type of choice
DEVICE = CL_DEVICE_TYPE_DEFAULT
export DEVICE

# Incase you need to rename the C++ compiler, you can
# do it in bulk here
CPPC = g++
export CPPC

ifndef CC
        CC = gcc
endif
export CC

.PHONY : $(CEXES) $(CPEXES)

all: $(CEXES) $(CPPEXES)

$(CEXES):
	$(MAKE) -C `dirname $@`

$(CPPEXES):
	$(MAKE) -C `dirname $@`

.PHONY : clean
clean:
	for e in $(CEXES) $(CPPEXES); do $(MAKE) -C `dirname $$e` clean; done
