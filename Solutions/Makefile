
# This makefile will produce all the C binaries
# in their respective directories

CEXES = Exercise04/C/vadd_chain Exercise05/C/vadd_abc \
		Exercise06/C/mult Exercise07/C/mult \
		Exercise08/C/pi_ocl

# Change this variable to specify the device type in all
# the Makefile to the OpenCL device type of choice
DEVICE = CL_DEVICE_TYPE_DEFAULT
export DEVICE

.PHONY : $(CEXES)

all: $(CEXES)

$(CEXES):
	$(MAKE) -C `dirname $@`

.PHONY : clean
clean:
	for e in $(CEXES); do $(MAKE) -C `dirname $$e` clean; done
