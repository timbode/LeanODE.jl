# One solve per method while the package is precompiled, so that the parts of
# a solve that do not depend on the right-hand side are compiled once, into
# the package image, and not at the first solve of every session. That first
# solve then takes about a third of the time.

"""
    precompile_lorenz!(du, u, p, t)

The Lorenz system with `σ = 10`, `ρ = 28`, `β = 8/3`, solved while the
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
    for alg in (Tsit5(), RK4(), VCABM())
        solve(ODEProblem(precompile_lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0)), alg)
        solve(ODEProblem(precompile_lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0)), alg; saveat = 0.5, reltol = 1.0e-6, abstol = 1.0e-6)
    end
end
