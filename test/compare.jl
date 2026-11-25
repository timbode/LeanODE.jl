# Solving one problem with LeanODE and with OrdinaryDiffEq, and comparing.

"""The OrdinaryDiffEq method that LeanODE's `alg` was copied from."""
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
the OrdinaryDiffEq method it was copied from, passing the same keywords to
both. OrdinaryDiffEq's warnings (for example on instability) are silenced.
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
    identical(lean, reference)

`true` if the two solutions agree bit for bit: the saved times and states
(compared with `isequal`, so `-0.0 ≠ 0.0` and NaN equals NaN), the numbers of
function evaluations, accepted and rejected steps, and the return code.
"""
function identical(lean, reference)
    return isequal(lean.t, reference.t) &&
        isequal(lean.u, reference.u) &&
        lean.stats.nf == reference.stats.nf &&
        lean.stats.naccept == reference.stats.naccept &&
        lean.stats.nreject == reference.stats.nreject &&
        Symbol(lean.retcode) == Symbol(reference.retcode)
end

"""
    describe_difference(lean, reference)

A one-line account of where two solutions differ, for test failure messages.
"""
function describe_difference(lean, reference)
    n = min(length(lean.t), length(reference.t))
    first_diff = findfirst(i -> !isequal(lean.t[i], reference.t[i]) || !isequal(lean.u[i], reference.u[i]), 1:n)
    return "saves $(length(lean.t)) vs $(length(reference.t)), first different save $(first_diff), " *
        "nf $(lean.stats.nf) vs $(reference.stats.nf), naccept $(lean.stats.naccept) vs $(reference.stats.naccept), " *
        "nreject $(lean.stats.nreject) vs $(reference.stats.nreject), retcode $(lean.retcode) vs $(reference.retcode)"
end
