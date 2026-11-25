# After OrdinaryDiffEqAdamsBashforthMoulton 1.11.0 (src/algorithms.jl,
# src/alg_utils.jl, src/adams_bashforth_moulton_caches.jl), MIT licence; see
# LICENSE.md. The methods are OrdinaryDiffEq's; the caches hold only what
# LeanODE's sweeps over the state need (adams_utils.jl).

"""
    VCABM3()

The third-order Adams–Bashforth–Moulton predictor–corrector method with
variable step size, as in OrdinaryDiffEq: two evaluations of `f` per step.
The first two steps are taken with `BS3`.
"""
struct VCABM3 <: LeanAlgorithm end

"""
    VCABM4()

The fourth-order Adams–Bashforth–Moulton predictor–corrector method with
variable step size, as in OrdinaryDiffEq: two evaluations of `f` per step.
The first three steps are taken with `RK4`.
"""
struct VCABM4 <: LeanAlgorithm end

"""
    VCABM5()

The fifth-order Adams–Bashforth–Moulton predictor–corrector method with
variable step size, as in OrdinaryDiffEq: two evaluations of `f` per step.
The first four steps are taken with `RK4`.
"""
struct VCABM5 <: LeanAlgorithm end

"""
    VCABM()

The Adams–Bashforth–Moulton predictor–corrector method with variable step
size and variable order, from 1 to 12, as in OrdinaryDiffEq: two evaluations
of `f` per step. It starts at order 1 and chooses the order by the rules of
Shampine's DDEABM, so it needs no other method for its first steps.
"""
struct VCABM <: LeanAlgorithm end

"""
    alg_order(::VCABM3)

VCABM3 has order 3, VCABM4 order 4 and VCABM5 order 5. For the
variable-order `VCABM` the value 1 is a placeholder, as in OrdinaryDiffEq;
its current order is `get_current_alg_order`.
"""
alg_order(alg::VCABM3) = 3
alg_order(alg::VCABM4) = 4
alg_order(alg::VCABM5) = 5
alg_order(alg::VCABM) = 1  #dummy value

"""
    isstandard(::VCABM)

`VCABM` uses the integral step-size controller (see `controller.jl`).
"""
isstandard(alg::VCABM) = true

"""
    get_current_alg_order(::VCABM, cache)

The order `VCABM` currently runs at.
"""
get_current_alg_order(alg::VCABM, cache) = cache.order

"""
    get_current_adaptive_order(::VCABM, cache)

The order that sets the exponent of `VCABM`'s step-size controller: its
current order.
"""
get_current_adaptive_order(alg::VCABM, cache) = cache.order

"""
    AdamsCache

Work arrays of an Adams method with variable step size, for orders up to
`max_order`.

# Fields
- `fsalfirst`, `k4`: `f(uprev)` and `f(u)`.
- `starter`: the cache of the Runge–Kutta method that takes the first steps
  of a fixed-order method; `nothing` for the variable-order method.
- `ϕstar_n`, `ϕstar_nm1`: the divided differences ϕ*ᵢ of this step and of the
  last one (see adams_utils.jl), `max_order + 1` arrays each. They change
  places after every accepted step.
- `ϕ_k`: the difference ϕₖ(n) of the current order.
- `ψ`: a block vector of length `BLOCK`, scratch space of the sweeps.
- `tmp`: scratch space for the initial step size.
- `dts`: the last step sizes, newest first.
- `β`, `c`, `g`: the coefficients of adams_utils.jl.
- `ξ`, `ξ0`: the sums of step sizes that `β_coefs!` ended with.
- `order`: the order of the next step, and `max_order` its bound.
- `step`: for a fixed-order method, one more than the number of steps
  attempted so far, up to `order`; below `order` the next step is a
  Runge–Kutta step.
"""
mutable struct AdamsCache{Alg, uType, Starter, T}
    fsalfirst::uType
    k4::uType
    starter::Starter
    ϕstar_n::Vector{uType}
    ϕstar_nm1::Vector{uType}
    ϕ_k::uType
    ψ::Vector{T}
    tmp::uType
    dts::Vector{Float64}
    β::Vector{Float64}
    c::Matrix{Float64}
    g::Vector{Float64}
    ξ::Float64
    ξ0::Float64
    order::Int
    max_order::Int
    step::Int
end

"""
    AdamsCache(alg, u, rate_prototype, starter, order, max_order)

Allocate the work arrays of the Adams method `alg` for states like `u`,
starting at `order`.
"""
function AdamsCache(alg::Alg, u, rate_prototype, starter, order, max_order) where {Alg}
    count = max_order + 1
    T = eltype(u)
    return AdamsCache{Alg, typeof(u), typeof(starter), T}(
        zero(rate_prototype), zero(rate_prototype), starter,
        [zero(rate_prototype) for _ in 1:count], [zero(rate_prototype) for _ in 1:count],
        zero(rate_prototype),
        zeros(T, BLOCK),
        zero(u),
        zeros(count), zeros(count), zeros(count, count), zeros(count),
        0.0, 0.0, order, max_order, 1
    )
end

"""
    VCABM3Cache, VCABM4Cache, VCABM5Cache, VCABMCache

The caches of the four Adams methods.
"""
const VCABM3Cache = AdamsCache{VCABM3}
const VCABM4Cache = AdamsCache{VCABM4}
const VCABM5Cache = AdamsCache{VCABM5}
const VCABMCache = AdamsCache{VCABM}

"""
    FixedOrderCache

The caches of VCABM3, VCABM4 and VCABM5.
"""
const FixedOrderCache = Union{VCABM3Cache, VCABM4Cache, VCABM5Cache}

"""
    alg_cache(alg, u, rate_prototype, uprev)

Allocate the work arrays of the Adams method `alg` for states like `u`.
VCABM3 starts with BS3 steps, VCABM4 and VCABM5 with RK4 steps; VCABM starts
at order 1 and may rise to 12.
"""
alg_cache(alg::VCABM3, u, rate_prototype, uprev) =
    AdamsCache(alg, u, rate_prototype, alg_cache(BS3(), u, rate_prototype, uprev), 3, 3)
alg_cache(alg::VCABM4, u, rate_prototype, uprev) =
    AdamsCache(alg, u, rate_prototype, alg_cache(RK4(), u, rate_prototype, uprev), 4, 4)
alg_cache(alg::VCABM5, u, rate_prototype, uprev) =
    AdamsCache(alg, u, rate_prototype, alg_cache(RK4(), u, rate_prototype, uprev), 5, 5)
alg_cache(alg::VCABM, u, rate_prototype, uprev) =
    AdamsCache(alg, u, rate_prototype, nothing, 1, 12)

"""
    get_fsalfirstlast(cache::AdamsCache)

The arrays that hold `f(uprev)` and `f(u)`: `fsalfirst` and `k4`.
"""
get_fsalfirstlast(cache::AdamsCache) = (cache.fsalfirst, cache.k4)

"""
    exchange_fsal!(integrator, cache::AdamsCache)

Exchange `fsalfirst` and `k4`, the arrays that hold `f(uprev)` and `f(u)`.
"""
function exchange_fsal!(integrator, cache::AdamsCache)
    cache.fsalfirst, cache.k4 = cache.k4, cache.fsalfirst
    integrator.fsalfirst = integrator.k[1] = cache.fsalfirst
    integrator.fsallast = integrator.k[2] = cache.k4
    return nothing
end
