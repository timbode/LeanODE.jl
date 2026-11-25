"""
    ReturnCode

The outcome of a solve. `ReturnCode.Success` means the integration reached
the end of the time span. The other values name the reason it stopped early:

- `MaxIters`: more than `maxiters` steps were attempted.
- `DtNaN`: the step size became NaN.
- `DtLessThanMin`: the step size fell to `dtmin`.
- `Unstable`: the state became infinite or NaN, or the step size fell below
  floating-point resolution of `t`.
- `Failure`: an OrdinaryDiffEq method, called through `solve` with SciMLBase
  loaded, stopped for a reason LeanODE does not name.

`Default` is the value during the solve. `Symbol(code)` gives the name, for
example `:Success`, as for OrdinaryDiffEq's return codes.
"""
module ReturnCode
@enum T Default Success MaxIters DtNaN DtLessThanMin Unstable ConvergenceFailure Failure
end

"""
    Stats

Work counters of a solve.

# Fields
- `nf::Int`: number of evaluations of the right-hand side `f`.
- `naccept::Int`: number of accepted steps.
- `nreject::Int`: number of rejected steps.
"""
mutable struct Stats
    nf::Int
    naccept::Int
    nreject::Int
end

"""
    Stats()

Return counters that are all zero.
"""
Stats() = Stats(0, 0, 0)

"""
    ODEProblem(f!, u0, tspan, p = nothing)

The initial value problem `du/dt = f(u, p, t)`, `u(tspan[1]) = u0`.

# Arguments
- `f!`: the right-hand side in place, called as `f!(du, u, p, t)`; it writes
  the derivative into `du`.
- `u0::Array`: the initial state, an array of `Float64` or `ComplexF64` of any
  dimension. The solver copies it and never changes it.
- `tspan`: `(t0, tf)`. The time runs backwards if `tf < t0`.
- `p`: parameters, passed to `f!` unchanged.

# Throws
- `ArgumentError` if `u0` is not an array of `Float64` or `ComplexF64`.
"""
struct ODEProblem{uType <: Array, P, F}
    f::F
    u0::uType
    tspan::Tuple{Float64, Float64}
    p::P

    function ODEProblem(f::F, u0::uType, tspan, p::P = nothing) where {F, uType, P}
        eltype(u0) in (Float64, ComplexF64) || throw(
            ArgumentError("u0 must be an Array of Float64 or ComplexF64, got $(typeof(u0))")
        )
        return new{uType, P, F}(f, u0, (Float64(tspan[1]), Float64(tspan[2])), p)
    end
end

"""
    LeanAlgorithm

Supertype of the integration methods (`Tsit5`, `RK4`, `BS3`, `VCABM`,
`VCABM3`, `VCABM4`, `VCABM5`).
"""
abstract type LeanAlgorithm end

"""
    ODESolution

The result of [`solve`](@ref).

# Fields
- `t::Vector{Float64}`: the saved times.
- `u::Vector`: the saved states; `u[i]` is the state at `t[i]`.
- `retcode::ReturnCode.T`: how the solve ended.
- `stats::Stats`: work counters.
- `prob::ODEProblem`: what was solved.
- `alg`: the method, a `LeanAlgorithm` or (through the SciMLBase extension) an
  OrdinaryDiffEq method.
"""
struct ODESolution{uType, Prob, Alg}
    t::Vector{Float64}
    u::Vector{uType}
    retcode::ReturnCode.T
    stats::Stats
    prob::Prob
    alg::Alg
end
