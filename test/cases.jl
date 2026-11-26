# The comparisons: every problem crossed with tolerances and ways of saving,
# then step-size options, early stops and edge cases, for both methods. Each
# case is one problem solved once by LeanODE and once by OrdinaryDiffEq with
# the same keywords; the two solutions must be identical.

"""Saving options for a time span: every step, the end only, and several kinds of `saveat`."""
function saving_options(tspan)
    t0, tf = tspan
    Δ = tf - t0
    return (
        every_step = (;),
        end_only = (; save_everystep = false),
        saveat_grid = (; saveat = collect(range(t0, tf; length = 11))),
        saveat_interior = (; saveat = t0 .+ Δ .* [0.137, 0.5, 0.9, 0.999]),
        saveat_number = (; saveat = abs(Δ) / 7),
        saveat_without_ends = (; saveat = t0 .+ Δ .* [0.25, 0.75], save_start = false, save_end = false),
    )
end

const TOLERANCES = (
    default = (;),
    loose = (; reltol = 1.0e-4, abstol = 1.0e-6),
    tight = (; reltol = 1.0e-10, abstol = 1.0e-10),
)

const ALGORITHMS = (
    LeanODE.Tsit5(), LeanODE.RK4(), LeanODE.BS3(),
    LeanODE.VCABM3(), LeanODE.VCABM4(), LeanODE.VCABM5(), LeanODE.VCABM(),
)

"""
    special_cases()

Cases beyond the grid, as `(label, problem, keywords, retcode)`. `retcode` is
the name of the return code the case is built to end with, or `nothing` if it
only has to match.
"""
function special_cases()
    P = PROBLEMS
    lorenz_short = merge(P.lorenz, (; tspan = (0.0, 2.0)))
    blowup = (f! = blowup!, u0 = [1.0], tspan = (0.0, 2.0), p = nothing)
    return [
        # step-size options
        ("fixed step", lorenz_short, (; adaptive = false, dt = 0.01), :Success),
        ("fixed step that does not divide the span", lorenz_short, (; adaptive = false, dt = 0.03, saveat = 0.5), :Success),
        ("fixed step, negative dt on a forward span", P.decay, (; adaptive = false, dt = -0.1), :Success),
        ("initial dt given", lorenz_short, (; dt = 1.0e-3), :Success),
        ("initial dt far too large", P.decay, (; dt = 100.0), :Success),
        ("dtmax", lorenz_short, (; dtmax = 0.05), :Success),
        ("relative tolerance only", lorenz_short, (; reltol = 1.0e-9), :Success),
        ("backwards, dtmax", P.decay_backwards, (; dtmax = 0.01), :Success),
        ("backwards, initial dt given", P.decay_backwards, (; dt = 1.0e-3), :Success),
        ("backwards, fixed step", P.decay_backwards, (; adaptive = false, dt = 0.03), :Success),
        # early stops
        ("maxiters = 50", P.lorenz, (; maxiters = 50), :MaxIters),
        ("NaN after t = 0.5", (f! = nan_after!, u0 = [1.0, 2.0], tspan = (0.0, 1.0), p = 0.5), (;), :Unstable),
        ("blow-up", blowup, (;), :Unstable),
        ("blow-up with dtmin", blowup, (; dtmin = 1.0e-3), :DtLessThanMin),
        ("blow-up of a complex state", merge(blowup, (; u0 = ComplexF64[1, 0.5])), (;), :Unstable),
        ("dt = NaN", P.decay, (; dt = NaN), :DtNaN),
        ("NaN in the initial state", merge(P.decay, (; u0 = [NaN, 1.0])), (;), :DtNaN),
        ("Inf in the initial state", merge(P.decay, (; u0 = [Inf, 1.0])), (;), :DtNaN),
        # early stops that save only the end: the state saved last is whatever the
        # final attempted step left behind. Tsit5 rejects its 14th attempted step
        # of this problem and RK4 its 21st, so each method stops once right after
        # a rejected step and once right after an accepted one.
        ("maxiters = 14, only the end saved", P.van_der_pol, (; maxiters = 14, save_everystep = false), :MaxIters),
        ("maxiters = 21, only the end saved", P.van_der_pol, (; maxiters = 21, save_everystep = false), :MaxIters),
        ("NaN after t = 0.5, only the end saved", (f! = nan_after!, u0 = [1.0, 2.0], tspan = (0.0, 1.0), p = 0.5), (; save_everystep = false), :Unstable),
        ("blow-up, only the end saved", blowup, (; save_everystep = false), :Unstable),
        ("blow-up with dtmin, only the end saved", blowup, (; dtmin = 1.0e-3, save_everystep = false), :DtLessThanMin),
        # unusual problems
        ("zero initial state", merge(P.decay, (; u0 = [0.0, 0.0])), (;), :Success),
        ("time span of length zero", merge(P.decay, (; tspan = (1.0, 1.0))), (;), :Success),
        # one representable number apart. An adaptive solve stops at once, because
        # no step size fits; with fixed steps, forwards is a single step of that
        # length and backwards across a power of two is a step too short to take,
        # which is accepted without being taken.
        ("time span of one ulp", merge(P.decay, (; tspan = (1.0, nextfloat(1.0)))), (;), :Unstable),
        ("time span of one ulp, backwards", merge(P.decay, (; tspan = (2.0, prevfloat(2.0)))), (;), :Unstable),
        ("time span of one ulp, fixed step", merge(P.decay, (; tspan = (1.0, nextfloat(1.0)))), (; adaptive = false, dt = 0.5), :Success),
        ("time span of one ulp, backwards, fixed step", merge(P.decay, (; tspan = (2.0, prevfloat(2.0)))), (; adaptive = false, dt = 0.5), :Success),
        ("time span of one ulp, backwards, fixed step, saveat", merge(P.decay, (; tspan = (2.0, prevfloat(2.0)))), (; adaptive = false, dt = 0.5, saveat = [prevfloat(2.0)]), :Success),
        ("empty state", merge(P.decay, (; u0 = Float64[])), (;), :Success),
        ("integer time span", merge(P.decay, (; tspan = (0, 2))), (;), :Success),
        ("three-dimensional state", merge(P.decay, (; u0 = reshape(collect(1.0:24.0), 2, 3, 4))), (;), :Success),
        ("complex vector of 21", merge(P.rotation, (; u0 = ComplexF64[i * cis(0.3i) for i in 1:21])), (; saveat = [0.1, 1.7]), :Success),
        ("time span not starting at zero", merge(P.lorenz, (; tspan = (3.5, 5.5))), (; saveat = 0.3), :Success),
        # unusual ways of saving
        ("saveat as a range", P.lorenz, (; saveat = 0.0:0.5:20.0), :Success),
        ("saveat times outside the span", P.decay, (; saveat = [-1.0, 0.5, 1.0, 5.0]), :Success),
        ("saveat unsorted with a duplicate", P.decay, (; saveat = [1.5, 0.5, 0.5, 1.0]), :Success),
        ("saveat and every step", P.van_der_pol, (; saveat = [0.3, 7.7], save_everystep = true), :Success),
        ("saveat with save_start", P.decay, (; saveat = [0.5], save_start = true), :Success),
        ("saveat with save_end", P.decay, (; saveat = [0.5], save_end = true), :Success),
        ("nothing saved", P.decay, (; save_everystep = false, save_start = false, save_end = false), :Success),
        ("save_end = false", P.decay, (; save_end = false), :Success),
        ("backwards, saveat as a number", P.decay_backwards, (; saveat = 0.3), :Success),
    ]
end

"""
    RETCODE_EXCEPTIONS

Special cases that end differently with one method, as
`(method, label) => return code`. Without step-size control, OrdinaryDiffEq's
VCABM stays at order 1 and evaluates `f` only at the predicted state, and
that is unstable for the Lorenz system at a step size of 0.03.
"""
const RETCODE_EXCEPTIONS = Dict(
    (LeanODE.VCABM, "fixed step that does not divide the span") => :Unstable,
)

"""
    comparison_cases()

All comparisons, as named tuples `(; group, label, problem, alg, keywords, retcode)`.
`group` is the testset the case is reported under; `retcode` is as in
[`special_cases`](@ref) unless `RETCODE_EXCEPTIONS` names another, and
`nothing` for the grid.
"""
function comparison_cases()
    cases = NamedTuple[]
    for alg in ALGORITHMS
        for (name, problem) in pairs(PROBLEMS), (tol_name, tol) in pairs(TOLERANCES),
                (save_name, save) in pairs(saving_options(problem.tspan))
            push!(cases, (; group = String(name), label = "$name, $tol_name, $save_name", problem, alg, keywords = merge(tol, save), retcode = nothing))
        end
        for (label, problem, keywords, retcode) in special_cases()
            expected = get(RETCODE_EXCEPTIONS, (typeof(alg), label), retcode)
            push!(cases, (; group = "special cases", label, problem, alg, keywords, retcode = expected))
        end
    end
    return cases
end

"""
    compare(case)

Solve one case with both packages. Returns `(; same, retcode_as_expected, rejections, detail)`:
whether the solutions are identical, whether OrdinaryDiffEq's return code is
the one the case was built for, OrdinaryDiffEq's number of rejected steps,
and for a mismatch a description of it.
"""
function compare(case)
    lean, reference = both_solutions(case.problem, case.alg; case.keywords...)
    same = identical(lean, reference)
    retcode_as_expected = case.retcode === nothing || Symbol(reference.retcode) == case.retcode
    detail = same ? "" : describe_difference(lean, reference)
    return (; same, retcode_as_expected, rejections = reference.stats.nreject, detail)
end
