# Copied from OrdinaryDiffEqTsit5 1.12.0 (src/tsit_caches.jl, src/tsit_perform_step.jl,
# src/interpolants.jl), MIT licence; see LICENSE.md. Only the in-place method
# for `Array` states is kept, without stage and step limiters. The broadcasts
# stay FastBroadcast's `@..` (vendored): Julia lowers `muladd` to contractable
# multiply and add, and whether the backend fuses them depends on the loop it
# sits in, so Base's `@.` would change the last bits.

"""
    Tsit5()

Tsitouras' 5(4) Runge–Kutta pair (Ch. Tsitouras, Comput. Math. Appl. 62,
2011), as in OrdinaryDiffEq: seven stages, first-same-as-last, with an
embedded fourth-order error estimate and a free fourth-order interpolant.
"""
struct Tsit5 <: LeanAlgorithm end

"""
    alg_order(alg)

Order of the method `alg`; it sets the controller's exponents and the
initial step size.
"""
alg_order(::Tsit5) = 5

"""
    Tsit5Cache

Work arrays of Tsit5: the stage derivatives `k1`–`k7`, the error estimate
`utilde`, a stage state `tmp`, and the scaled error `atmp`. `thread` selects
FastBroadcast's serial kernel. `u` and `uprev`
are the integrator's arrays.
"""
struct Tsit5Cache{uType}
    u::uType
    uprev::uType
    k1::uType
    k2::uType
    k3::uType
    k4::uType
    k5::uType
    k6::uType
    k7::uType
    utilde::uType
    tmp::uType
    atmp::uType
    thread::FastBroadcast.False
end

"""
    alg_cache(alg, u, rate_prototype, uprev)

Allocate the work arrays of the method `alg` for states like `u`;
derivatives are shaped like `rate_prototype`.
"""
function alg_cache(::Tsit5, u, rate_prototype, uprev)
    k1 = zero(rate_prototype)
    k2 = zero(rate_prototype)
    k3 = zero(rate_prototype)
    k4 = zero(rate_prototype)
    k5 = zero(rate_prototype)
    k6 = zero(rate_prototype)
    k7 = zero(rate_prototype)
    utilde = zero(u)
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    tmp = zero(u)
    return Tsit5Cache(u, uprev, k1, k2, k3, k4, k5, k6, k7, utilde, tmp, atmp, FastBroadcast.False())
end

"""
    get_fsalfirstlast(cache)

The arrays that hold `f(uprev)` and `f(u)`: for Tsit5 the first and the last
stage, `k1` and `k7`.
"""
get_fsalfirstlast(cache::Tsit5Cache) = (cache.k1, cache.k7)

"""
    initialize!(integrator, cache)

Point the interpolation stages at the cache and evaluate `f(u0)` into
`fsalfirst`.
"""
function initialize!(integrator, cache::Tsit5Cache)
    resize!(integrator.k, 7)
    integrator.k[1] = cache.k1
    integrator.k[2] = cache.k2
    integrator.k[3] = cache.k3
    integrator.k[4] = cache.k4
    integrator.k[5] = cache.k5
    integrator.k[6] = cache.k6
    integrator.k[7] = cache.k7
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1
    return nothing
end

"""
    perform_step!(integrator, cache)

Take one step from `(t, uprev)` with step size `dt`, writing the new state
into `u`, the stages into the cache, and (if adaptive) the scaled error
estimate into `integrator.EEst`.
"""
@muladd function perform_step!(integrator, cache::Tsit5Cache)
    (; t, dt, uprev, u, f, p) = integrator
    T = constvalue(recursive_unitless_bottom_eltype(u))
    T2 = constvalue(typeof(one(t)))
    @OnDemandTableauExtract Tsit5ConstantCacheActual T T2
    (; k1, k2, k3, k4, k5, k6, k7, utilde, tmp, atmp, thread) = cache
    a = dt * a21
    @.. broadcast = false thread = thread tmp = uprev + a * k1
    f(k2, tmp, p, t + c1 * dt)
    @.. broadcast = false thread = thread tmp = uprev + dt * (a31 * k1 + a32 * k2)
    f(k3, tmp, p, t + c2 * dt)
    @.. broadcast = false thread = thread tmp = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    f(k4, tmp, p, t + c3 * dt)
    @.. broadcast = false thread = thread tmp = uprev +
        dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    f(k5, tmp, p, t + c4 * dt)
    @.. broadcast = false thread = thread tmp = uprev +
        dt * (
        a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 +
            a65 * k5
    )
    f(k6, tmp, p, t + dt)
    @.. broadcast = false thread = thread u = uprev +
        dt * (
        a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 +
            a75 * k5 + a76 * k6
    )
    f(k7, u, p, t + dt)
    integrator.stats.nf += 6
    if integrator.opts.adaptive
        @.. broadcast = false thread = thread utilde = dt * (
            btilde1 * k1 + btilde2 * k2 +
                btilde3 * k3 + btilde4 * k4 +
                btilde5 * k5 + btilde6 * k6 +
                btilde7 * k7
        )
        calculate_residuals!(
            atmp, utilde, uprev, u, integrator.opts.abstol,
            integrator.opts.reltol, ode_default_norm, t
        )
        integrator.EEst = ode_default_norm(atmp, t)
    end
    return nothing
end

"""
    interpolant!(out, Θ, dt, y₀, y₁, k, cache)

Write into `out` the state at `tprev + Θ dt` within the last step, from
`y₀ = uprev`, `y₁ = u` and the stages `k`. For Tsit5 this is Tsitouras' free
fourth-order interpolant, `y₀ + dt Σᵢ bᵢ(Θ) kᵢ`. Returns `out`.
"""
@muladd function interpolant!(
        out::Array, Θ, dt, y₀, y₁, k, cache::Tsit5Cache
    )
    var"#T#" = constvalue(recursive_unitless_bottom_eltype(y₁))
    r11, r12, r13, r14, r22, r23, r24, r32, r33, r34, r42, r43, r44, r52, r53, r54, r62, r63, r64, r72, r73, r74 = Tsit5Interp(var"#T#")
    Θ² = Θ * Θ
    b1Θ = Θ * @evalpoly(Θ, r11, r12, r13, r14)
    b2Θ = Θ² * @evalpoly(Θ, r22, r23, r24)
    b3Θ = Θ² * @evalpoly(Θ, r32, r33, r34)
    b4Θ = Θ² * @evalpoly(Θ, r42, r43, r44)
    b5Θ = Θ² * @evalpoly(Θ, r52, r53, r54)
    b6Θ = Θ² * @evalpoly(Θ, r62, r63, r64)
    b7Θ = Θ² * @evalpoly(Θ, r72, r73, r74)
    @inbounds @simd ivdep for i in eachindex(out)
        out[i] = y₀[i] +
            dt * (
            k[1][i] * b1Θ + k[2][i] * b2Θ + k[3][i] * b3Θ + k[4][i] * b4Θ +
                k[5][i] * b5Θ + k[6][i] * b6Θ + k[7][i] * b7Θ
        )
    end
    return out
end
