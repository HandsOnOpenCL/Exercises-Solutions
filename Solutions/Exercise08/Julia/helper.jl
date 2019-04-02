function error{T}(Mdim::Int, Ndim::Int, Pdim::Int, C::Array{T})
    cval  = float32(Pdim * AVAL * BVAL)
    errsq = float32(0.0)
    for i in 1:Ndim
        for j in 1:Mdim
            err = C[(i-1)*Ndim+j] - cval
            errsq += err^2
        end
    end
    return errsq
end

function results{T}(Mdim::Int, Ndim::Int, Pdim::Int, C::Array{T}, run_time)
    mflops = 2.0 * Mdim * Ndim * Pdim/(1000000.0* run_time)
    println("$run_time seconds at $mflops MFLOPS")
    errsq = error(Mdim, Ndim, Pdim, C)
    if isnan(errsq) || errsq > TOL
        println("Errors in multiplication: $errsq")
    end
end
