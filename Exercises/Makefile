
# This makefile will produce all the C binaries
# in their respective directories

CEXES = Exercise01/C/DeviceInfo Exercise02/C/vadd \
		Exercise06/C/mult Exercise08/C/pi

.PHONY : $(CEXES)

all: $(CEXES)

$(CEXES):
	$(MAKE) -C `dirname $@`

.PHONY : clean
clean:
	rm -f $(CEXES)
	rm -rf *.o
