
# This makefile will produce all the C binaries
# in their respective directories

CEXES = Exercise01/C/DeviceInfo Exercise02/C/vadd \
		Exercise04/C/vadd Exercise05/C/vadd \
		Exercise06/C/mult Exercise09/C/pi \
		Exercise13/C/gameoflife

CPPEXES = Exercise01/Cpp/DeviceInfo Exercise03/Cpp/vadd \
		Exercise04/Cpp/vadd Exercise05/Cpp/vadd \
		Exercise06/Cpp/mult Exercise09/Cpp/pi

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

.PHONY : $(CEXES) $(CPPEXES)

all: $(CEXES) $(CPPEXES)

$(CEXES):
	$(MAKE) -C `dirname $@`

$(CPPEXES):
	$(MAKE) -C `dirname $@`

.PHONY : clean
clean:
	for e in $(CEXES) $(CPPEXES); do $(MAKE) -C `dirname $$e` clean; done
