# LeanODE precompiles what OrdinaryDiffEqTsit5 1.12.0 precompiles (the workload in
# its src/OrdinaryDiffEqTsit5.jl): one Tsit5 solve of the Lorenz system on a
# `Vector{Float64}`, without `saveat`. Nothing more, and nothing less.
#
# Julia's ahead-of-time compiler rounds some multiply-adds differently from its
# just-in-time compiler (see test/runtests.jl). The broadcast kernels are
# functions of their own that do not depend on the right-hand side, so the ones
# this solve compiles into the package image are then used by every Tsit5 solve
# on a `Vector{Float64}`, exactly as in OrdinaryDiffEq. Precompiling a different
# set would make an ordinary session differ from OrdinaryDiffEq in the last
# bits. The precompiled kernels are also faster: Tsit5 on three components
# takes about a third less time per step.

"""
    precompile_lorenz!(du, u, p, t)

The Lorenz system with `σ = 10`, `ρ = 28`, `β = 8/3`, solved once while the
package is precompiled.
"""
function precompile_lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
    return nothing
end

# `jl_generating_output` is 1 only while Julia writes a package image.
if ccall(:jl_generating_output, Cint, ()) == 1
    solve(ODEProblem(precompile_lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0)), Tsit5())
end
