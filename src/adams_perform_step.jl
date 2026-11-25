# After OrdinaryDiffEqAdamsBashforthMoulton 1.11.0
# (src/adams_bashforth_moulton_perform_step.jl), MIT licence; see LICENSE.md.
# The steps do what upstream's in-place methods of VCABM3, VCABM4, VCABM5 and
# VCABM do, in the same order and with the same arithmetic per component, in
# the sweeps of adams_utils.jl. Upstream writes the three fixed-order methods
# out separately; here they are one function, and the few places where
# upstream's methods differ from each other are small functions of their own.

"""
    initialize!(integrator, cache::AdamsCache)

Point the two interpolation derivatives at `fsalfirst` and `fsallast`, and
evaluate `f(u0)` into `fsalfirst`.
"""
function initialize!(integrator, cache::AdamsCache)
    resize!(integrator.k, 2)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # pre-start FSAL
    integrator.stats.nf += 1
    return nothing
end

"""
    exchange_differences!(cache)

Make this step's divided differences `ϕstar_n` the last step's, `ϕstar_nm1`.
"""
function exchange_differences!(cache)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    return nothing
end

# ------------------------------------------------- VCABM3, VCABM4, VCABM5

"""
    push_step_size!(dts, dt, count)

Enter `dt` as the newest of the first `count` step sizes in `dts`, moving
the others back by one; the oldest of them is dropped.
"""
function push_step_size!(dts, dt, count)
    @inbounds for i in count:-1:2
        dts[i] = dts[i - 1]
    end
    @inbounds dts[1] = dt
    return nothing
end

"""
    restore_step_sizes!(cache, oldest)

Undo `push_step_size!` after a step that will be rejected; `oldest` is the
step size it dropped. For VCABM5, OrdinaryDiffEq leaves the third entry as
it is, and so does this method.
"""
function restore_step_sizes!(cache::Union{VCABM3Cache, VCABM4Cache}, oldest)
    (; dts, order) = cache
    @inbounds for i in 1:(order - 1)
        dts[i] = dts[i + 1]
    end
    @inbounds dts[order] = oldest
    return nothing
end

function restore_step_sizes!(cache::VCABM5Cache, oldest)
    (; dts) = cache
    @inbounds begin
        dts[1] = dts[2]
        dts[2] = dts[3]
        dts[4] = dts[5]
        dts[5] = oldest
    end
    return nothing
end

"""
    difference_order(cache, k)

The order up to which a fixed-order method forms its divided differences in
step number `k`: `k` itself for VCABM3 and VCABM4, always 5 for VCABM5.
"""
difference_order(cache::Union{VCABM3Cache, VCABM4Cache}, k) = k
difference_order(cache::VCABM5Cache, k) = 5

"""
    starting_step!(integrator, cache)

Take one of the first steps of a fixed-order method with its Runge–Kutta
method, and copy `f(u)` into `integrator.fsallast`: BS3 for VCABM3, RK4 for
VCABM4 and VCABM5.
"""
function starting_step!(integrator, cache::VCABM3Cache)
    bs3cache = cache.starter
    perform_step!(integrator, bs3cache)
    integrator.fsallast .= bs3cache.k4
    return nothing
end

function starting_step!(integrator, cache::Union{VCABM4Cache, VCABM5Cache})
    rk4cache = cache.starter
    rk4cache.fsalfirst .= integrator.fsalfirst
    perform_step!(integrator, rk4cache)
    integrator.fsallast .= rk4cache.k
    return nothing
end

"""
    perform_step!(integrator, cache::FixedOrderCache)

Take one step of VCABM3, VCABM4 or VCABM5 from `(t, uprev)` with step size
`dt` into `u`. The first `order - 1` steps are Runge–Kutta steps, which also
supply the error estimate; every attempt among them counts, whether accepted
or not. An Adams step whose error estimate exceeds 1 will be rejected, so
the step sizes are restored and the function returns without evaluating
`f(u)`.
"""
function perform_step!(integrator, cache::FixedOrderCache)
    (; t, dt, uprev, u, f, p) = integrator
    (; abstol, reltol, adaptive) = integrator.opts
    (; k4, dts, order) = cache
    k1 = integrator.fsalfirst
    k = cache.step
    oldest = dts[order]
    push_step_size!(dts, dt, k)
    k < order && (cache.step += 1)
    cache.ξ, cache.ξ0 = β_coefs!(cache, difference_order(cache, k))
    if k < order
        predict!(u, uprev, k1, cache, difference_order(cache, k), 0)
        starting_step!(integrator, cache)
    else
        g_coefs!(cache, k + 1)
        predict!(u, uprev, k1, cache, k, k - 1)
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
        estimate, _, _ = correct!(u, uprev, k4, cache, k, abstol, reltol, false)
        if adaptive
            integrator.EEst = estimate
            if integrator.EEst > one(integrator.EEst)
                restore_step_sizes!(cache, oldest)
                return nothing
            end
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
    end
    exchange_differences!(cache)
    return nothing
end

# ------------------------------------------------------------------ VCABM

"""
    choose_order!(integrator, cache, k, one_lower, two_lower)

Choose the order of the next step of the variable-order method after an
accepted step at order `k ≥ 3`, from the error estimates of the neighbouring
orders: `one_lower` and `two_lower` for `k - 1` and `k - 2`, and the
estimate for `k + 1`, which this function computes. The order falls if both
lower orders would have been at least as accurate as `k`, and rises if the
higher one would have been more accurate. After a rise the error estimate is
set to 1, which keeps the step size for the next step.
"""
function choose_order!(integrator, cache, k, one_lower, two_lower)
    (; t, dt, uprev, u) = integrator
    (; abstol, reltol) = integrator.opts
    one_higher = higher_order_error!(cache, cache.k4, uprev, u, k, dt, abstol, reltol)
    if max(two_lower, one_lower) <= integrator.EEst
        cache.order = k - 1
    elseif one_higher < integrator.EEst
        cache.order = min(k + 1, cache.max_order)
        integrator.EEst = one(integrator.EEst)    # for keeping the stepsize constant in the next step
    end
    return nothing
end

"""
    perform_step!(integrator, cache::VCABMCache)

Take one step of the variable-order VCABM from `(t, uprev)` with step size
`dt` into `u`, at the order `cache.order`, and choose the order of the next
step. During the first four steps and below order 3 the order rises by one
per step, up to 3; later `choose_order!` decides.

As in OrdinaryDiffEq, the predictor has at least one term, also at order 1,
and without step-size control neither `f(u)` is evaluated nor the order
changed.
"""
function perform_step!(integrator, cache::VCABMCache)
    (; t, dt, uprev, u, f, p) = integrator
    (; abstol, reltol, adaptive) = integrator.opts
    (; k4, dts) = cache
    k1 = integrator.fsalfirst
    step = integrator.iter
    k = cache.order
    oldest = dts[13]
    push_step_size!(dts, dt, 13)
    cache.ξ, cache.ξ0 = β_coefs!(cache, k)
    g_coefs!(cache, k + 1)
    predict!(u, uprev, k1, cache, k, max(k - 1, 1))
    f(k4, u, p, t + dt)
    integrator.stats.nf += 1
    rising = step <= 4 || k < 3
    estimate, one_lower, two_lower = correct!(u, uprev, k4, cache, k, abstol, reltol, adaptive && !rising)
    if adaptive
        integrator.EEst = estimate
        if integrator.EEst > one(integrator.EEst)
            @inbounds for i in 1:12
                dts[i] = dts[i + 1]
            end
            @inbounds dts[13] = oldest
            return nothing
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
        if rising
            cache.order = min(k + 1, 3)
        else
            choose_order!(integrator, cache, k, one_lower, two_lower)
        end
    end
    exchange_differences!(cache)
    return nothing
end
