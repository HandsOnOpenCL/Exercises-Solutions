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
#          Ported to Julia by Jake Bolewski, Nov 2013

const num_steps = 100000000

println("Doing $num_steps")

const step = 1.0/num_steps

start_time = time()

# global variables are slow in julia, so we wrap in a function
pi_sum() = begin
    integral_sum = 0.0
    for i in 1:num_steps
        x = (i - 0.5) * step
        integral_sum += 4.0 / (1.0 + x * x)
    end
    integral_sum
end

est_pi = step * pi_sum()

run_time = time() - start_time;

println("\npi with $num_steps steps is $est_pi in $run_time seconds\n")
