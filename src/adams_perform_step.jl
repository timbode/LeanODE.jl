# Copied from OrdinaryDiffEqAdamsBashforthMoulton 1.11.0
# (src/adams_bashforth_moulton_perform_step.jl), MIT licence; see LICENSE.md.
# Only the in-place methods of VCABM3, VCABM4, VCABM5 and VCABM are kept, and
# the reset after a callback changed the state is deleted. The three
# fixed-order methods differ only slightly, but each stays as upstream wrote
# it, so that the results stay the same in every bit.
#
# A step, in the notation of adams_utils.jl: the step size enters `dts`, the
# divided differences `ϕ_n`, `ϕstar_n` are formed from `f(uprev)`, the
# predictor is evaluated, `f` at the predictor gives `ϕ_np1`, and the
# corrector adds its last term. The error estimate is the difference between
# the correctors of this order and the next, `(gₖ₊₁ - gₖ) ϕₖ₊₁(n + 1)`. A
# step whose estimate exceeds 1 will be rejected, so `dts` is restored and
# the function returns early. Otherwise `f` is evaluated at the new state
# and `ϕstar_n` becomes `ϕstar_nm1` for the next step.

"""
    initialize!(integrator, cache::VariableCoefficientCache)

Point the two interpolation derivatives at `fsalfirst` and `fsallast`, and
evaluate `f(u0)` into `fsalfirst`.
"""
function initialize!(integrator, cache::VariableCoefficientCache)
    resize!(integrator.k, 2)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # pre-start FSAL
    integrator.stats.nf += 1
    return nothing
end

# VCABM3

"""
    perform_step!(integrator, cache::VCABM3Cache)

Take one VCABM3 step from `(t, uprev)` with step size `dt` into `u`. The
first two steps are BS3 steps, which also supply the error estimate; every
attempt among them counts, whether accepted or not.
"""
@muladd function perform_step!(integrator, cache::VCABM3Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (;
        k4, dts, g, ϕstar_n, ϕ_np1, ϕstar_nm1, order, atmp, utilde, bs3cache,
        thread,
    ) = cache
    k1 = integrator.fsalfirst
    k = cache.step
    tmp = dts[3]
    if k == 1
        dts[1] = dt
        cache.step += 1
    elseif k == 2
        dts[2] = dts[1]
        dts[1] = dt
        cache.step += 1
    else
        dts[3] = dts[2]
        dts[2] = dts[1]
        dts[1] = dt
    end
    ϕ_and_ϕstar!(cache, k1, k)
    if k == 1 || k == 2
        perform_step!(integrator, bs3cache)
        (; k4) = bs3cache
        integrator.fsallast .= k4
    else
        g_coefs!(cache, k + 1)
        @.. broadcast = false thread = thread u = uprev
        for i in 1:(k - 1)
            @.. broadcast = false thread = thread u += g[i] * ϕstar_n[i]
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
        ϕ_np1!(cache, k4, k + 1)
        @.. broadcast = false thread = thread u += g[end - 1] * ϕ_np1[end - 1]
        if integrator.opts.adaptive
            @.. broadcast = false thread = thread utilde = (g[end] - g[end - 1]) * ϕ_np1[end]
            calculate_residuals!(
                atmp, utilde, uprev, u, integrator.opts.abstol,
                integrator.opts.reltol, ode_default_norm, t
            )
            integrator.EEst = ode_default_norm(atmp, t)
            if integrator.EEst > one(integrator.EEst)
                for i in 1:2
                    dts[i] = dts[i + 1]
                end
                dts[3] = tmp
                return nothing
            end
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
    end
    cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
    return nothing
end

# VCABM4

"""
    perform_step!(integrator, cache::VCABM4Cache)

Take one VCABM4 step from `(t, uprev)` with step size `dt` into `u`. The
first three steps are RK4 steps.
"""
@muladd function perform_step!(integrator, cache::VCABM4Cache)
    (; t, dt, uprev, u, f, p) = integrator
    (;
        k4, dts, g, ϕstar_n, ϕ_np1, ϕstar_nm1, order, atmp, utilde, rk4cache,
        thread,
    ) = cache
    k1 = integrator.fsalfirst
    k = cache.step
    tmp = dts[4]
    if k == 1
        dts[1] = dt
        cache.step += 1
    elseif k == 2
        dts[2] = dts[1]
        dts[1] = dt
        cache.step += 1
    elseif k == 3
        dts[3] = dts[2]
        dts[2] = dts[1]
        dts[1] = dt
        cache.step += 1
    else
        dts[4] = dts[3]
        dts[3] = dts[2]
        dts[2] = dts[1]
        dts[1] = dt
    end
    ϕ_and_ϕstar!(cache, k1, k)
    if k == 1 || k == 2 || k == 3
        rk4cache.fsalfirst .= k1
        perform_step!(integrator, rk4cache)
        integrator.fsallast .= rk4cache.k
    else
        g_coefs!(cache, k + 1)
        @.. broadcast = false thread = thread u = uprev
        for i in 1:(k - 1)
            @.. broadcast = false thread = thread u += g[i] * ϕstar_n[i]
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
        ϕ_np1!(cache, k4, k + 1)
        @.. broadcast = false thread = thread u += g[end - 1] * ϕ_np1[end - 1]
        if integrator.opts.adaptive
            @.. broadcast = false thread = thread utilde = (g[end] - g[end - 1]) * ϕ_np1[end]
            calculate_residuals!(
                atmp, utilde, uprev, u, integrator.opts.abstol,
                integrator.opts.reltol, ode_default_norm, t
            )
            integrator.EEst = ode_default_norm(atmp, t)
            if integrator.EEst > one(integrator.EEst)
                for i in 1:3
                    dts[i] = dts[i + 1]
                end
                dts[4] = tmp
                return nothing
            end
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
    end
    cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
    return nothing
end

# VCABM5

"""
    perform_step!(integrator, cache::VCABM5Cache)

Take one VCABM5 step from `(t, uprev)` with step size `dt` into `u`. The
first four steps are RK4 steps.
"""
@muladd function perform_step!(integrator, cache::VCABM5Cache)
    @inbounds begin
        (; t, dt, uprev, u, f, p) = integrator
        (;
            k4, dts, g, ϕ_n, ϕ_np1, ϕstar_n, ϕstar_nm1,
            order, atmp, utilde, rk4cache, thread,
        ) = cache
        k1 = integrator.fsalfirst
        k = cache.step
        tmp = dts[5]
        if k == 5
            dts[5] = dts[4]
            dts[4] = dts[3]
            dts[3] = dts[2]
            dts[2] = dts[1]
            dts[1] = dt
        elseif k == 1
            dts[1] = dt
            cache.step += 1
        elseif k == 2
            dts[2] = dts[1]
            dts[1] = dt
            cache.step += 1
        elseif k == 3
            dts[3] = dts[2]
            dts[2] = dts[1]
            dts[1] = dt
            cache.step += 1
        elseif k == 4
            dts[4] = dts[3]
            dts[3] = dts[2]
            dts[2] = dts[1]
            dts[1] = dt
            cache.step += 1
        end
        ϕ_and_ϕstar!(cache, k1, 5)
        if k <= 4
            rk4cache.fsalfirst .= k1
            perform_step!(integrator, rk4cache)
            integrator.fsallast .= rk4cache.k
        else
            g_coefs!(cache, 6)
            @.. broadcast = false thread = thread u = muladd(g[1], ϕstar_n[1], uprev)
            @.. broadcast = false thread = thread u = muladd(g[2], ϕstar_n[2], u)
            @.. broadcast = false thread = thread u = muladd(g[3], ϕstar_n[3], u)
            @.. broadcast = false thread = thread u = muladd(g[4], ϕstar_n[4], u)
            f(k4, u, p, t + dt)
            integrator.stats.nf += 1
            ϕ_np1!(cache, k4, 6)
            @.. broadcast = false thread = thread u = muladd(g[6 - 1], ϕ_np1[6 - 1], u)
            if integrator.opts.adaptive
                @.. broadcast = false thread = thread utilde = (g[6] - g[6 - 1]) * ϕ_np1[end]
                calculate_residuals!(
                    atmp, utilde, uprev, u, integrator.opts.abstol,
                    integrator.opts.reltol, ode_default_norm,
                    t
                )
                integrator.EEst = ode_default_norm(atmp, t)
                if integrator.EEst > one(integrator.EEst)
                    dts[1] = dts[2]
                    dts[2] = dts[3]
                    dts[4] = dts[5]
                    dts[5] = tmp
                    return nothing
                end
            end
            cache.ϕstar_nm1[1] .= ϕstar_n[1]
            cache.ϕstar_nm1[2] .= ϕstar_n[2]
            cache.ϕstar_nm1[3] .= ϕstar_n[3]
            cache.ϕstar_nm1[4] .= ϕstar_n[4]
            f(k4, u, p, t + dt)
            integrator.stats.nf += 1
        end
        cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
        return nothing
    end # inbounds
end

# VCABM

"""
    perform_step!(integrator, cache::VCABMCache)

Take one step of the variable-order VCABM from `(t, uprev)` with step size
`dt` into `u`, at the order `cache.order`, and choose the order of the next
step. During the first four steps and below order 3 the order rises by one
per step, up to 3. Later, the error estimates of the orders one and two
below and one above the current order decide: the order falls if both lower
orders would have been at least as accurate, and rises if the higher one
would have been more accurate. After a rise the estimate is set to 1, which
keeps the step size for the next step.
"""
@muladd function perform_step!(integrator, cache::VCABMCache)
    @inbounds begin
        (; t, dt, uprev, u, f, p) = integrator
        (;
            k4, dts, g, ϕ_n, ϕ_np1, ϕstar_n, ϕstar_nm1, order, max_order, utilde,
            utildem2, utildem1, utildep1, atmp, atmpm1, atmpm2, atmpp1, thread,
        ) = cache
        k1 = integrator.fsalfirst
        step = integrator.iter
        k = order
        tmp = dts[13]
        for i in 12:-1:1
            dts[i + 1] = dts[i]
        end
        dts[1] = dt
        ϕ_and_ϕstar!(cache, k1, k)
        g_coefs!(cache, k + 1)
        # unroll the predictor once
        @.. broadcast = false thread = thread u = muladd(g[1], ϕstar_n[1], uprev)
        for i in 2:(k - 1)
            @.. broadcast = false thread = thread u = muladd(g[i], ϕstar_n[i], u)
        end
        f(k4, u, p, t + dt)
        integrator.stats.nf += 1
        ϕ_np1!(cache, k4, k + 1)
        @.. broadcast = false thread = thread u = muladd(g[k], ϕ_np1[k], u)
        if integrator.opts.adaptive
            @.. broadcast = false thread = thread utilde = (g[k + 1] - g[k]) * ϕ_np1[k + 1]
            calculate_residuals!(
                atmp, utilde, uprev, u, integrator.opts.abstol,
                integrator.opts.reltol, ode_default_norm, t
            )
            integrator.EEst = ode_default_norm(atmp, t)
            if integrator.EEst > one(integrator.EEst)
                for i in 1:12
                    dts[i] = dts[i + 1]
                end
                dts[13] = tmp
                return nothing
            end
            f(k4, u, p, t + dt)
            integrator.stats.nf += 1
            if step <= 4 || order < 3
                cache.order = min(order + 1, 3)
            else
                # @.. broadcast=false thread=thread utildem2 = dt * γstar[(k-2)+1] * ϕ_np1[k-1]
                @.. broadcast = false thread = thread utildem2 = (g[k - 1] - g[k - 2]) *
                    ϕ_np1[k - 1]
                # @.. broadcast=false thread=thread utildem1 = dt * γstar[(k-1)+1] * ϕ_np1[k]
                @.. broadcast = false thread = thread utildem1 = (g[k] - g[k - 1]) * ϕ_np1[k]
                expand_ϕ_and_ϕstar!(cache, k + 1)
                ϕ_np1!(cache, k4, k + 2)
                @.. broadcast = false thread = thread utildep1 = dt * γstar[(k + 1) + 1] *
                    ϕ_np1[k + 2]
                calculate_residuals!(
                    atmpm2, utildem2, uprev, u, integrator.opts.abstol,
                    integrator.opts.reltol, ode_default_norm,
                    t
                )
                calculate_residuals!(
                    atmpm1, utildem1, uprev, u, integrator.opts.abstol,
                    integrator.opts.reltol, ode_default_norm,
                    t
                )
                calculate_residuals!(
                    atmpp1, utildep1, uprev, u, integrator.opts.abstol,
                    integrator.opts.reltol, ode_default_norm,
                    t
                )
                errm2 = ode_default_norm(atmpm2, t)
                errm1 = ode_default_norm(atmpm1, t)
                errp1 = ode_default_norm(atmpp1, t)
                if max(errm2, errm1) <= integrator.EEst
                    cache.order = order - 1
                elseif errp1 < integrator.EEst
                    cache.order = min(order + 1, max_order)
                    integrator.EEst = one(integrator.EEst)    # for keeping the stepsize constant in the next step
                end
            end
        end
        cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
        return nothing
    end # inbounds
end
