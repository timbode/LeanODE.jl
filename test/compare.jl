# Solving one problem with LeanODE and with OrdinaryDiffEq, and measuring how
# far the two solutions are apart.

"""The OrdinaryDiffEq method that LeanODE's `alg` implements."""
reference_alg(::LeanODE.Tsit5) = OrdinaryDiffEqTsit5.Tsit5()
reference_alg(::LeanODE.RK4) = OrdinaryDiffEqLowOrderRK.RK4()
reference_alg(::LeanODE.BS3) = OrdinaryDiffEqLowOrderRK.BS3()
reference_alg(::LeanODE.VCABM3) = OrdinaryDiffEqAdamsBashforthMoulton.VCABM3()
reference_alg(::LeanODE.VCABM4) = OrdinaryDiffEqAdamsBashforthMoulton.VCABM4()
reference_alg(::LeanODE.VCABM5) = OrdinaryDiffEqAdamsBashforthMoulton.VCABM5()
reference_alg(::LeanODE.VCABM) = OrdinaryDiffEqAdamsBashforthMoulton.VCABM()

"""
    both_solutions(problem, alg; kwargs...)

Solve `problem` (a named tuple from `PROBLEMS`) with LeanODE's `alg` and with
the same method of OrdinaryDiffEq, passing the same keywords to both.
OrdinaryDiffEq's warnings (for example on instability) are silenced.
"""
function both_solutions(problem, alg; kwargs...)
    (; f!, u0, tspan, p) = problem
    lean = LeanODE.solve(LeanODE.ODEProblem(f!, u0, tspan, p), alg; kwargs...)
    reference = with_logger(NullLogger()) do
        SciMLBase.solve(SciMLBase.ODEProblem(f!, u0, tspan, p), reference_alg(alg); kwargs...)
    end
    return lean, reference
end

"""
    largest_magnitude(states)

The largest finite `|x|` over all components of all arrays in `states`; zero
if there is none.
"""
function largest_magnitude(states)
    largest = 0.0
    for state in states, x in state
        isfinite(x) && (largest = max(largest, abs(x)))
    end
    return largest
end

"""
    relative_distance(lean_states, reference_states)

The largest difference between corresponding components of two lists of
arrays, divided by the largest magnitude in the reference (undivided if that
is zero). Components that are equal, including equal infinities and NaNs,
count as zero; a NaN or infinity on one side only gives `Inf`.
"""
function relative_distance(lean_states, reference_states)
    distance = 0.0
    for (a, b) in zip(lean_states, reference_states), (x, y) in zip(a, b)
        isequal(x, y) && continue
        difference = abs(x - y)
        distance = max(distance, isfinite(difference) ? difference : Inf)
    end
    scale = largest_magnitude(reference_states)
    return scale == 0 ? distance : distance / scale
end

"""
    tolerance_distance(lean, reference, abstol, reltol)

The largest difference between two solutions at the saved times they have in
common, in units of the tolerance: `|Δuᵢ| / (abstol + reltol |uᵢ|)`,
maximised over components and common times. Returns the distance and the
number of common times.
"""
function tolerance_distance(lean, reference, abstol, reltol)
    distance = 0.0
    common = 0
    for (i, t) in enumerate(reference.t)
        j = findfirst(==(t), lean.t)
        j === nothing && continue
        common += 1
        for (x, y) in zip(lean.u[j], reference.u[i])
            isequal(x, y) && continue
            distance = max(distance, abs(x - y) / (abstol + reltol * abs(y)))
        end
    end
    return distance, common
end

"""
    same_counts(lean, reference)

`true` if both solves evaluated `f` equally often and accepted and rejected
equally many steps.
"""
function same_counts(lean, reference)
    return lean.stats.nf == reference.stats.nf &&
        lean.stats.naccept == reference.stats.naccept &&
        lean.stats.nreject == reference.stats.nreject
end

"""
    states_along(problem, alg, count; kwargs...)

`count` points `(t, u, dt)` spread along OrdinaryDiffEq's solution of
`problem` (solved with the keywords `kwargs`): a time, the state there, and
the size of the step the solver took next.
"""
function states_along(problem, alg, count; kwargs...)
    (; f!, u0, tspan, p) = problem
    reference = SciMLBase.solve(SciMLBase.ODEProblem(f!, u0, tspan, p), reference_alg(alg); kwargs...)
    steps = length(reference.t) - 1
    chosen = unique(round.(Int, range(1, steps; length = min(count, steps))))
    return [(; t = reference.t[i], u = reference.u[i], dt = reference.t[i + 1] - reference.t[i]) for i in chosen]
end

"""
    one_step(problem, alg, point; kwargs...)

Take one adaptive step of size `point.dt` from the state `point.u` at time
`point.t`, with LeanODE and with OrdinaryDiffEq. Returns for each a named
tuple `(; t, u, error_estimate, next_dt)`: the time and state after the step,
the scaled error estimate and the step size proposed for the next step.
"""
function one_step(problem, alg, point; kwargs...)
    (; f!, p) = problem
    tspan = (point.t, problem.tspan[2])
    dt = abs(point.dt)
    lean = LeanODE.init(LeanODE.ODEProblem(f!, point.u, tspan, p), alg; dt, kwargs...)
    LeanODE.loopheader!(lean)
    LeanODE.perform_step!(lean)
    LeanODE.loopfooter!(lean)
    reference = SciMLBase.init(SciMLBase.ODEProblem(f!, point.u, tspan, p), reference_alg(alg); dt, kwargs...)
    SciMLBase.step!(reference)
    return (; lean.t, u = copy(lean.u), error_estimate = lean.EEst, next_dt = lean.dtpropose),
        (; reference.t, u = copy(reference.u), error_estimate = reference.EEst, next_dt = SciMLBase.get_proposed_dt(reference))
end

"""
    lean_step!(integrator)

Advance a LeanODE integrator by one accepted step, repeating rejected
attempts, as `SciMLBase.step!` does for OrdinaryDiffEq's.
"""
function lean_step!(integrator)
    while true
        LeanODE.loopheader!(integrator)
        LeanODE.perform_step!(integrator)
        LeanODE.loopfooter!(integrator)
        integrator.accept_step && break
    end
    LeanODE.handle_tstop!(integrator)
    return nothing
end

"""
    current_order(integrator)

The order a method currently runs at, for the methods that store one in
their cache (the Adams methods); 0 for the others.
"""
current_order(integrator) = hasproperty(integrator.cache, :order) ? integrator.cache.order : 0

"""
    first_steps(problem, alg, count; kwargs...)

Take the first `count` steps of `problem` with LeanODE and with
OrdinaryDiffEq, one at a time, from integrators set up with the keywords
`kwargs`. Returns for each step a pair of named tuples
`(; t, u, error_estimate, next_dt, order, rejections)`: the time and state
after the step, the scaled error estimate, the step size proposed for the
next step, the order of the next step, and the number of rejected attempts
so far.
"""
function first_steps(problem, alg, count; kwargs...)
    (; f!, u0, tspan, p) = problem
    lean = LeanODE.init(LeanODE.ODEProblem(f!, u0, tspan, p), alg; kwargs...)
    reference = SciMLBase.init(SciMLBase.ODEProblem(f!, u0, tspan, p), reference_alg(alg); kwargs...)
    steps = []
    for _ in 1:count
        lean_step!(lean)
        SciMLBase.step!(reference)
        push!(
            steps,
            (
                (; lean.t, u = copy(lean.u), error_estimate = lean.EEst, next_dt = lean.dtpropose, order = current_order(lean), rejections = lean.stats.nreject),
                (; reference.t, u = copy(reference.u), error_estimate = reference.EEst, next_dt = SciMLBase.get_proposed_dt(reference), order = current_order(reference), rejections = reference.stats.nreject),
            )
        )
    end
    return steps
end
