# After OrdinaryDiffEqLowOrderRK 1.13.0 (src/low_order_rk_caches.jl,
# src/fixed_timestep_perform_step.jl), MIT licence; see LICENSE.md. Only the
# in-place method for `Array` states is kept, without stage and step limiters.
# The method and its error estimate are OrdinaryDiffEq's; the loops over the
# state are LeanODE's own (kernels.jl, hermite.jl).

"""
    RK4()

The classical fourth-order Runge–Kutta method, as in OrdinaryDiffEq, with
adaptive step size: the error is estimated from the residual of the cubic Hermite
interpolant at two interior points of each step (Shampine, "Solving ODEs and
DDEs with residual control", Appl. Numer. Math. 52, 2005), which costs two
extra evaluations of `f` per step.
"""
struct RK4 <: LeanAlgorithm end

"""
    alg_order(::RK4)

RK4 has order 4.
"""
alg_order(::RK4) = 4

"""
    RK4Cache

Work arrays of RK4: the stage derivatives `fsalfirst` (`k₁`), `k₂`, `k₃`,
`k₄` and `k` (`f` at the new state), and a stage state `tmp`. The struct is
mutable because `fsalfirst` and `k` change places after every accepted step
(see `exchange_fsal!`).
"""
mutable struct RK4Cache{uType}
    fsalfirst::uType
    k₂::uType
    k₃::uType
    k₄::uType
    k::uType
    tmp::uType
end

"""
    alg_cache(::RK4, u, rate_prototype, uprev)

Allocate the work arrays of RK4 for states like `u`.
"""
function alg_cache(::RK4, u, rate_prototype, uprev)
    k₁ = zero(rate_prototype)
    k₂ = zero(rate_prototype)
    k₃ = zero(rate_prototype)
    k₄ = zero(rate_prototype)
    k = zero(rate_prototype)
    tmp = zero(u)
    return RK4Cache(k₁, k₂, k₃, k₄, k, tmp)
end

"""
    get_fsalfirstlast(cache::RK4Cache)

The arrays that hold `f(uprev)` and `f(u)`: `fsalfirst` and `k`.
"""
get_fsalfirstlast(cache::RK4Cache) = (cache.fsalfirst, cache.k)

"""
    exchange_fsal!(integrator, cache::RK4Cache)

Exchange `fsalfirst` and `k`, the arrays that hold `f(uprev)` and `f(u)`.
"""
function exchange_fsal!(integrator, cache::RK4Cache)
    cache.fsalfirst, cache.k = cache.k, cache.fsalfirst
    integrator.fsalfirst = integrator.k[1] = cache.fsalfirst
    integrator.fsallast = integrator.k[2] = cache.k
    return nothing
end

"""
    initialize!(integrator, cache::RK4Cache)

Point the two interpolation derivatives at the cache and evaluate `f(u0)`
into `fsalfirst`.
"""
function initialize!(integrator, cache::RK4Cache)
    integrator.fsalfirst = cache.fsalfirst
    integrator.fsallast = cache.k
    resize!(integrator.k, 2)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # pre-start FSAL
    integrator.stats.nf += 1
    return nothing
end

"""
    perform_step!(integrator, cache::RK4Cache)

Take one RK4 step from `(t, uprev)` with step size `dt` into `u`, and
evaluate `f(u)` into `cache.k`. If adaptive, store the error estimate in
`integrator.EEst` (see `residual_estimate!`).
"""
function perform_step!(integrator, cache::RK4Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (; tmp, fsalfirst, k₂, k₃, k₄, k) = cache
    k₁ = fsalfirst
    halfdt = dt / 2
    combine!(tmp, uprev, halfdt, k₁)
    f(k₂, tmp, p, t + halfdt)
    combine!(tmp, uprev, halfdt, k₂)
    f(k₃, tmp, p, t + halfdt)
    combine!(tmp, uprev, dt, k₃)
    f(k₄, tmp, p, t + dt)
    combine!(u, uprev, dt / 6, k₁, dt / 3, k₂, dt / 3, k₃, dt / 6, k₄)
    f(k, u, p, t + dt)
    integrator.stats.nf += 4
    if integrator.opts.adaptive
        integrator.EEst = residual_estimate!(integrator, cache)
        integrator.stats.nf += 2
    end
    return nothing
end

"""
    RESIDUAL_NODES

The two positions within a step, `1/2 ∓ √3/6`, at which RK4 measures the
residual of the Hermite interpolant.
"""
const RESIDUAL_NODES = (1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / 6)

"""
    residual_estimate!(integrator, cache::RK4Cache)

The scaled error estimate of the RK4 step just taken (Shampine, 2005): 2.1342
times the larger of the two residual norms of the cubic Hermite interpolant at
`RESIDUAL_NODES`. Evaluates `f` twice and overwrites the stage arrays `tmp`,
`k₂`, `k₃` and `k₄`, which the step no longer needs.
"""
function residual_estimate!(integrator, cache::RK4Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (; abstol, reltol) = integrator.opts
    slope₀, slope₁ = cache.fsalfirst, cache.k # f(uprev) and f(u)
    state₁, state₂, f₁, f₂ = cache.tmp, cache.k₄, cache.k₂, cache.k₃
    σ₁, σ₂ = RESIDUAL_NODES
    hermite_pair!(state₁, state₂, σ₁, σ₂, dt, uprev, u, slope₀, slope₁)
    f(f₁, state₁, p, t + σ₁ * dt)
    f(f₂, state₂, p, t + σ₂ * dt)
    e₁, e₂ = hermite_residual_norms(
        σ₁, σ₂, dt, uprev, u, slope₀, slope₁, f₁, f₂, abstol, reltol
    )
    return 2.1342 * max(e₁, e₂)
end
