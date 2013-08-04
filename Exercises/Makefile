
# This makefile will produce all the C binaries
# in their respective directories

CEXES = Exercise01/C/DeviceInfo Exercise02/C/vadd \
		Exercise04/C/vadd Exercise05/C/vadd \
		Exercise06/C/mult Exercise08/C/pi

# Change this variable to specify the device type in all
# the Makefile to the OpenCL device type of choice
DEVICE = CL_DEVICE_TYPE_DEFAULT
export DEVICE

# Check our platform and make sure we define the APPLE variable
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	APPLE = ''
	export APPLE
endif

.PHONY : $(CEXES)

all: $(CEXES)

$(CEXES):
	$(MAKE) -C `dirname $@`

.PHONY : clean
clean:
	for e in $(CEXES); do $(MAKE) -C `dirname $$e` clean; done
