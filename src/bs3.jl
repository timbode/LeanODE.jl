# Copied from OrdinaryDiffEqLowOrderRK 1.13.0 (src/low_order_rk_tableaus.jl,
# src/low_order_rk_caches.jl, src/low_order_rk_perform_step.jl), MIT licence;
# see LICENSE.md. Only the in-place method for `Array` states is kept, without
# stage and step limiters; the broadcasts stay `@..`, as in tsit5.jl.

"""
    BS3()

Bogacki and Shampine's 3(2) Runge–Kutta pair (P. Bogacki and L. F. Shampine,
Appl. Math. Lett. 2, 1989), as in OrdinaryDiffEq: three evaluations of `f`
per step, first-same-as-last, with an embedded second-order error estimate.
States between steps come from cubic Hermite interpolation.
"""
struct BS3 <: LeanAlgorithm end

"""
    alg_order(::BS3)

BS3 has order 3.
"""
alg_order(::BS3) = 3

"""
    BS3ConstantCache{T, T2}

The coefficients of the Bogacki–Shampine pair: the stage weights `a21`–`a43`
and the error weights `btilde1`–`btilde4` (type `T`), and the nodes `c1`, `c2`
(type `T2`, the time type). The third-order solution weights equal
`a41`–`a43` (the method is first-same-as-last).
"""
struct BS3ConstantCache{T, T2}
    a21::T
    a32::T
    a41::T
    a42::T
    a43::T
    c1::T2
    c2::T2
    btilde1::T
    btilde2::T
    btilde3::T
    btilde4::T
end

"""
    BS3ConstantCache(T, T2)

The Bogacki–Shampine coefficients as `Float64` literals, converted to `T`
and `T2`.
"""
function BS3ConstantCache(T::Type{<:CompiledFloats}, T2::Type{<:CompiledFloats})
    a21 = convert(T, 0.5)
    a32 = convert(T, 0.75)
    a41 = convert(T, 0.2222222222222222)
    a42 = convert(T, 0.3333333333333333)
    a43 = convert(T, 0.4444444444444444)
    c1 = convert(T2, 0.5)
    c2 = convert(T2, 0.75)
    btilde1 = convert(T, 0.06944444444444445)
    btilde2 = convert(T, -0.08333333333333333)
    btilde3 = convert(T, -0.1111111111111111)
    btilde4 = convert(T, 0.125)
    return BS3ConstantCache(a21, a32, a41, a42, a43, c1, c2, btilde1, btilde2, btilde3, btilde4)
end

"""
    BS3Cache

Work arrays of BS3: the stage derivatives `fsalfirst` (`k1`), `k2`, `k3` and
`k4` (`f` at the new state), the error estimate `utilde`, a stage state
`tmp`, the scaled error `atmp`, and the coefficients `tab`. `thread` selects
FastBroadcast's serial kernel. The struct is mutable because `fsalfirst` and
`k4` change places after every accepted step (see `exchange_fsal!`).
"""
mutable struct BS3Cache{uType, TabType}
    fsalfirst::uType
    k2::uType
    k3::uType
    k4::uType
    utilde::uType
    tmp::uType
    atmp::uType
    tab::TabType
    thread::FastBroadcast.False
end

"""
    alg_cache(::BS3, u, rate_prototype, uprev)

Allocate the work arrays of BS3 for states like `u`.
"""
function alg_cache(::BS3, u, rate_prototype, uprev)
    tab = BS3ConstantCache(constvalue(recursive_unitless_bottom_eltype(u)), Float64)
    k1 = zero(rate_prototype)
    k2 = zero(rate_prototype)
    k3 = zero(rate_prototype)
    k4 = zero(rate_prototype)
    utilde = zero(u)
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    tmp = zero(u)
    return BS3Cache(k1, k2, k3, k4, utilde, tmp, atmp, tab, FastBroadcast.False())
end

"""
    get_fsalfirstlast(cache::BS3Cache)

The arrays that hold `f(uprev)` and `f(u)`: `fsalfirst` and `k4`.
"""
get_fsalfirstlast(cache::BS3Cache) = (cache.fsalfirst, cache.k4)

"""
    exchange_fsal!(integrator, cache::BS3Cache)

Exchange `fsalfirst` and `k4`, the arrays that hold `f(uprev)` and `f(u)`.
"""
function exchange_fsal!(integrator, cache::BS3Cache)
    cache.fsalfirst, cache.k4 = cache.k4, cache.fsalfirst
    integrator.fsalfirst = integrator.k[1] = cache.fsalfirst
    integrator.fsallast = integrator.k[2] = cache.k4
    return nothing
end

"""
    initialize!(integrator, cache::BS3Cache)

Point the two interpolation derivatives at `fsalfirst` and `fsallast`, and
evaluate `f(u0)` into `fsalfirst`.
"""
function initialize!(integrator, cache::BS3Cache)
    resize!(integrator.k, 2)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1
    return nothing
end

"""
    perform_step!(integrator, cache::BS3Cache)

Take one BS3 step from `(t, uprev)` with step size `dt` into `u`, and
evaluate `f(u)` into `cache.k4`. If adaptive, store the scaled error estimate
in `integrator.EEst`. The first stage derivative is read from
`integrator.fsalfirst`, so the function also serves a method that uses BS3
for its first steps and keeps its own array for `f(uprev)`.
"""
@muladd function perform_step!(integrator, cache::BS3Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (; k2, k3, k4, utilde, tmp, atmp, thread) = cache
    (; a21, a32, a41, a42, a43, c1, c2, btilde1, btilde2, btilde3, btilde4) = cache.tab
    k1 = integrator.fsalfirst
    a1 = dt * a21
    @.. broadcast = false thread = thread tmp = uprev + a1 * k1
    f(k2, tmp, p, t + c1 * dt)
    a2 = dt * a32
    @.. broadcast = false thread = thread tmp = uprev + a2 * k2
    f(k3, tmp, p, t + c2 * dt)
    @.. broadcast = false thread = thread u = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    f(k4, u, p, t + dt)
    integrator.stats.nf += 3
    if integrator.opts.adaptive
        @.. broadcast = false thread = thread utilde = dt * (
            btilde1 * k1 + btilde2 * k2 +
                btilde3 * k3 + btilde4 * k4
        )
        calculate_residuals!(
            atmp, utilde, uprev, u, integrator.opts.abstol,
            integrator.opts.reltol, ode_default_norm, t
        )
        integrator.EEst = ode_default_norm(atmp, t)
    end
    return nothing
end
