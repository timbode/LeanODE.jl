# After OrdinaryDiffEqLowOrderRK 1.13.0 (src/low_order_rk_tableaus.jl,
# src/low_order_rk_caches.jl, src/low_order_rk_perform_step.jl), MIT licence;
# see LICENSE.md. Only the in-place method for `Array` states is kept, without
# stage and step limiters. The method and its coefficients are
# OrdinaryDiffEq's; the loops over the state are LeanODE's own (kernels.jl).

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
`k4` (`f` at the new state), and a stage state `tmp`. The struct is mutable
because `fsalfirst` and `k4` change places after every accepted step (see
`exchange_fsal!`).
"""
mutable struct BS3Cache{uType}
    fsalfirst::uType
    k2::uType
    k3::uType
    k4::uType
    tmp::uType
end

"""
    alg_cache(::BS3, u, rate_prototype, uprev)

Allocate the work arrays of BS3 for states like `u`.
"""
function alg_cache(::BS3, u, rate_prototype, uprev)
    return BS3Cache(
        zero(rate_prototype), zero(rate_prototype), zero(rate_prototype),
        zero(rate_prototype), zero(u)
    )
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
function perform_step!(integrator, cache::BS3Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (; k2, k3, k4, tmp) = cache
    (; a21, a32, a41, a42, a43, c1, c2, btilde1, btilde2, btilde3, btilde4) = BS3ConstantCache(Float64, Float64)
    k1 = integrator.fsalfirst
    combine!(tmp, uprev, dt * a21, k1)
    f(k2, tmp, p, t + c1 * dt)
    combine!(tmp, uprev, dt * a32, k2)
    f(k3, tmp, p, t + c2 * dt)
    combine!(u, uprev, dt * a41, k1, dt * a42, k2, dt * a43, k3)
    f(k4, u, p, t + dt)
    integrator.stats.nf += 3
    if integrator.opts.adaptive
        integrator.EEst = error_norm(
            uprev, u, integrator.opts.abstol, integrator.opts.reltol,
            dt * btilde1, k1, dt * btilde2, k2, dt * btilde3, k3, dt * btilde4, k4
        )
    end
    return nothing
end
