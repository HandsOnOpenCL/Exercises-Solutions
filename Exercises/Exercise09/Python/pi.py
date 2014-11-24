#
# This program will numerically compute the integral of
#
#                4/(1+x*x)
#
# from 0 to 1.  The value of this integral is pi -- which
# is great since it gives us an easy way to check the answer.
#
# This the original sequential program.
#
# History: Written in C by Tim Mattson, 11/99
#          Ported to Python by Tom Deakin, July 2013
#

from time import time

num_steps = 100000000

print "\nNote: Wanted to do", num_steps, "steps, but this is very slow in Python."

num_steps = 1000000

print "Doing", num_steps, "steps instead."

integral_sum = 0.0

step = 1.0/num_steps

start_time = time()

for i in range(1,num_steps):
    x = (i-0.5)*step
    integral_sum += 4.0/(1.0+x*x)

pi = step * integral_sum

run_time = time() - start_time;

print "\npi with", num_steps, "steps is", pi, "in", run_time, "seconds\n"

