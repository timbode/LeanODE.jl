# Copied from OrdinaryDiffEqLowOrderRK 1.13.0 (src/low_order_rk_caches.jl,
# src/fixed_timestep_perform_step.jl), MIT licence; see LICENSE.md. Only the
# in-place method for `Array` states is kept, without stage and step limiters;
# the broadcasts stay `@..`, as in tsit5.jl.

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
`k₄` and `k` (`f` at the new state), a stage state `tmp`, and the scaled
error `atmp`. `thread` selects FastBroadcast's serial kernel. The struct is
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
    atmp::uType
    thread::FastBroadcast.False
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
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    return RK4Cache(k₁, k₂, k₃, k₄, k, tmp, atmp, FastBroadcast.False())
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

Take one RK4 step from `(t, uprev)` with step size `dt` into `u`. If adaptive,
estimate the error from the defect of the Hermite interpolant at the interior
points `σ₁,₂ = 1/2 ∓ √3/6` and store it, scaled by 2.1342, in
`integrator.EEst`.
"""
@muladd function perform_step!(integrator, cache::RK4Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (; tmp, fsalfirst, k₂, k₃, k₄, k, atmp, thread) = cache
    k₁ = fsalfirst
    halfdt = dt / 2
    ttmp = t + halfdt
    @.. broadcast = false thread = thread tmp = uprev + halfdt * k₁
    f(k₂, tmp, p, ttmp)
    @.. broadcast = false thread = thread tmp = uprev + halfdt * k₂
    f(k₃, tmp, p, ttmp)
    @.. broadcast = false thread = thread tmp = uprev + dt * k₃
    f(k₄, tmp, p, t + dt)
    @.. broadcast = false thread = thread u = uprev + (dt / 6) * (2 * (k₂ + k₃) + (k₁ + k₄))
    f(k, u, p, t + dt)
    integrator.stats.nf += 4
    if integrator.opts.adaptive
        # Shampine Solving ODEs and DDEs with Residual Control Estimate
        k₅ = k
        _p = k₂
        pprime = k₃ # Alias some cache arrays
        # one(t) so that types are correct but unitless
        σ₁ = one(t) * (1 // 2) - sqrt(one(t) * 3) / 6
        σ₂ = one(t) * (1 // 2) + sqrt(one(t) * 3) / 6
        @.. broadcast = false thread = thread tmp = (1 - σ₁) * uprev + σ₁ * u +
            σ₁ * (σ₁ - 1) *
            (
            (1 - 2σ₁) * (u - uprev) +
                (σ₁ - 1) * dt * k₁ +
                σ₁ * dt * k₅
        )
        @.. broadcast = false thread = thread pprime = k₁ +
            σ₁ *
            (
            -4 * dt * k₁ - 2 * dt * k₅ - 6 * uprev +
                σ₁ *
                (
                3 * dt * k₁ + 3 * dt * k₅ + 6 * uprev -
                    6 * u
            ) +
                6 * u
        ) / dt
        f(_p, tmp, p, t + σ₁ * dt)
        @.. broadcast = false thread = thread tmp = dt * (_p - pprime)
        calculate_residuals!(
            atmp, tmp, uprev, u, integrator.opts.abstol,
            integrator.opts.reltol, ode_default_norm, t
        )
        e1 = ode_default_norm(atmp, t)
        @.. broadcast = false thread = thread tmp = (1 - σ₂) * uprev + σ₂ * u +
            σ₂ * (σ₂ - 1) *
            (
            (1 - 2σ₂) * (u - uprev) +
                (σ₂ - 1) * dt * k₁ +
                σ₂ * dt * k₅
        )
        @.. broadcast = false thread = thread pprime = k₁ +
            σ₂ *
            (
            -4 * dt * k₁ - 2 * dt * k₅ - 6 * uprev +
                σ₂ *
                (
                3 * dt * k₁ + 3 * dt * k₅ + 6 * uprev -
                    6 * u
            ) +
                6 * u
        ) / dt
        f(_p, tmp, p, t + σ₂ * dt)
        @.. broadcast = false thread = thread tmp = dt * (_p - pprime)
        calculate_residuals!(
            atmp, tmp, uprev, u, integrator.opts.abstol,
            integrator.opts.reltol, ode_default_norm, t
        )
        e2 = ode_default_norm(atmp, t)
        integrator.EEst = convert(typeof(one(t)), 2.1342) * max(e1, e2)
        integrator.stats.nf += 2
    end
    return nothing
end
