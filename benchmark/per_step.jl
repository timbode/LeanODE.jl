# Time per step of LeanODE and of OrdinaryDiffEq, on the same problems with the
# same method and options, after both have compiled the solve. Run from the
# repository root:
#
#     julia --project=benchmark benchmark/per_step.jl
#
# Prints one tab-separated line per case: problem, components, method, saving,
# steps of each package, seconds per step of each, and the ratio LeanODE /
# OrdinaryDiffEq. Then, per method and saving, the smallest and largest ratio
# over the problems. Takes about three minutes.

using BenchmarkTools, Printf
import LeanODE, SciMLBase, OrdinaryDiffEqTsit5, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqAdamsBashforthMoulton

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
    oscillators!(du, u, ω, t)

Independent harmonic oscillators `x'' = -ω² x`, stored as `(x₁, v₁, x₂, v₂, …)`;
the angular frequencies are `ω`.
"""
function oscillators!(du, u, ω, t)
    @inbounds for i in eachindex(ω)
        du[2i - 1] = u[2i]
        du[2i] = -ω[i]^2 * u[2i - 1]
    end
    return nothing
end

"""
    oscillator_problem(n)

`n ÷ 2` oscillators with frequencies spread over `[1, 2]`, started at
`x = 1, v = 0`, on `t ∈ [0, 10]`: `(f!, u0, tspan, p)`.
"""
function oscillator_problem(n)
    m = n ÷ 2
    ω = [1 + i / m for i in 1:m]
    u0 = repeat([1.0, 0.0], m)
    return (oscillators!, u0, (0.0, 10.0), ω)
end

const PROBLEMS = [
    ("Lorenz", 3, (lorenz!, [1.0, 0.0, 0.0], (0.0, 10.0), nothing)),
    ("oscillators", 100, oscillator_problem(100)),
    ("oscillators", 10_000, oscillator_problem(10_000)),
]

const METHODS = [
    (LeanODE.Tsit5(), OrdinaryDiffEqTsit5.Tsit5()),
    (LeanODE.RK4(), OrdinaryDiffEqLowOrderRK.RK4()),
    (LeanODE.BS3(), OrdinaryDiffEqLowOrderRK.BS3()),
    (LeanODE.VCABM3(), OrdinaryDiffEqAdamsBashforthMoulton.VCABM3()),
    (LeanODE.VCABM5(), OrdinaryDiffEqAdamsBashforthMoulton.VCABM5()),
    (LeanODE.VCABM(), OrdinaryDiffEqAdamsBashforthMoulton.VCABM()),
]

const SAVING = [("every step", (;)), ("end only", (; save_everystep = false))]

"""
    steps(solution)

Accepted plus rejected steps of a solution.
"""
steps(solution) = solution.stats.naccept + solution.stats.nreject

"""
    RATIOS

The ratios of time per step, LeanODE / OrdinaryDiffEq, by method and saving.
"""
const RATIOS = Dict{Tuple{Symbol, String}, Vector{Float64}}()

println("problem\tcomponents\tmethod\tsaving\tsteps (LeanODE)\tsteps (OrdinaryDiffEq)\ts per step (LeanODE)\ts per step (OrdinaryDiffEq)\tratio")
for (name, n, (f!, u0, tspan, p)) in PROBLEMS, (lean_alg, ode_alg) in METHODS, (saving, keywords) in SAVING
    lean_problem = LeanODE.ODEProblem(f!, u0, tspan, p)
    ode_problem = SciMLBase.ODEProblem(f!, u0, tspan, p)
    lean_steps = steps(LeanODE.solve(lean_problem, lean_alg; keywords...))
    ode_steps = steps(SciMLBase.solve(ode_problem, ode_alg; keywords...))
    lean_time = @belapsed LeanODE.solve($lean_problem, $lean_alg; $keywords...) seconds = 2
    ode_time = @belapsed SciMLBase.solve($ode_problem, $ode_alg; $keywords...) seconds = 2
    lean_per_step, ode_per_step = lean_time / lean_steps, ode_time / ode_steps
    method = nameof(typeof(lean_alg))
    push!(get!(RATIOS, (method, saving), Float64[]), lean_per_step / ode_per_step)
    println(join((name, n, method, saving, lean_steps, ode_steps,
        @sprintf("%.3g", lean_per_step), @sprintf("%.3g", ode_per_step), @sprintf("%.2f", lean_per_step / ode_per_step)), "\t"))
    flush(stdout)
end

println("\nmethod\tsaving\tsmallest ratio\tlargest ratio")
for (lean_alg, _) in METHODS, (saving, _) in SAVING
    ratios = RATIOS[(nameof(typeof(lean_alg)), saving)]
    println(join((nameof(typeof(lean_alg)), saving, @sprintf("%.2f", minimum(ratios)), @sprintf("%.2f", maximum(ratios))), "\t"))
end
