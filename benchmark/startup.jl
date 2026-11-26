# Time to load LeanODE or OrdinaryDiffEq, and to the first solve after loading.
# Run from the repository root:
#
#     julia --project=benchmark benchmark/startup.jl
#
# Each measurement runs in a fresh process, seven times; the script prints the
# medians. Run once before measuring, so that every package is precompiled.
# The script loads nothing itself, so that the processes start alike.

"""
    middle(values)

The median of an odd number of values.
"""
middle(values) = sort(values)[(length(values) + 1) ÷ 2]

"""
    lorenz!(du, u, p, t)

The Lorenz system with σ = 10, ρ = 28, β = 8/3.
"""
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
    return nothing
end

"""
    PACKAGE_OF

The package `using` loads for each method: LeanODE's, and the OrdinaryDiffEq
subpackage that has the method.
"""
const PACKAGE_OF = Dict(
    ("LeanODE", "Tsit5") => :LeanODE,
    ("LeanODE", "VCABM") => :LeanODE,
    ("OrdinaryDiffEq", "Tsit5") => :OrdinaryDiffEqTsit5,
    ("OrdinaryDiffEq", "VCABM") => :OrdinaryDiffEqAdamsBashforthMoulton,
)

"""
    measure_once(package, method)

In this process: load the package that has `method`, then solve the Lorenz
system on `t ∈ [0, 10]` with it. Prints the seconds for both, tab-separated.
"""
function measure_once(package, method)
    name = PACKAGE_OF[(package, method)]
    load = @elapsed @eval using $name
    mod = Base.invokelatest(getfield, Main, name)
    problem = Base.invokelatest(getfield(mod, :ODEProblem), lorenz!, [1.0, 0.0, 0.0], (0.0, 10.0))
    alg = Base.invokelatest(getfield(mod, Symbol(method)))
    first_solve = @elapsed Base.invokelatest(getfield(mod, :solve), problem, alg)
    println(load, "\t", first_solve)
    return nothing
end

"""
    measure(package, method; rounds = 7)

Run `measure_once` in `rounds` fresh processes and return the medians of the
load time and of the time of the first solve, in seconds. `rounds` must be
odd.
"""
function measure(package, method; rounds = 7)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) $package $method`
    times = [parse.(Float64, split(readchomp(command), '\t')) for _ in 1:rounds]
    return middle(first.(times)), middle(last.(times))
end

if length(ARGS) == 2
    measure_once(ARGS...)
else
    println("package\tmethod\tload (s)\tfirst solve (s)\ttogether (s)")
    for method in ("Tsit5", "VCABM"), package in ("LeanODE", "OrdinaryDiffEq")
        load, first_solve = measure(package, method)
        println(join((package, method, round(load; sigdigits = 2), round(first_solve; sigdigits = 2), round(load + first_solve; sigdigits = 2)), "\t"))
    end
end
