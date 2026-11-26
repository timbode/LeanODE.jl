# OrdinaryDiffEq's in-place step of the variable-order VCABM, the reference for
# LeanODE's VCABM in the comparison. runtests.jl evaluates this file inside
# OrdinaryDiffEqAdamsBashforthMoulton, where it replaces the method of the
# same signature for the test session; the installed package is not changed.
#
# Copied from OrdinaryDiffEqAdamsBashforthMoulton.jl 1.11.0,
# src/adams_bashforth_moulton_perform_step.jl, with one change, marked below.
# In 1.11.0 the predictor's first term is also present at order 1, and the
# corrector adds its term to the predictor, so that an order-1 step advances by
# twice the slope. At order 1 the corrector now starts from `uprev`, as in
# https://github.com/SciML/OrdinaryDiffEq.jl/pull/5056 and in LeanODE.

@muladd function perform_step!(integrator, cache::VCABMCache, repeat_step = false)
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
        OrdinaryDiffEqCore.increment_nf!(integrator.stats, 1)
        ϕ_np1!(cache, k4, k + 1)
        corrector_start = k == 1 ? uprev : u    # the change: `u` in 1.11.0
        @.. broadcast = false thread = thread u = muladd(g[k], ϕ_np1[k], corrector_start)
        if integrator.opts.adaptive
            @.. broadcast = false thread = thread utilde = (g[k + 1] - g[k]) * ϕ_np1[k + 1]
            calculate_residuals!(
                atmp, utilde, uprev, u, integrator.opts.abstol,
                integrator.opts.reltol, integrator.opts.internalnorm, t
            )
            integrator.EEst = integrator.opts.internalnorm(atmp, t)
            if integrator.EEst > one(integrator.EEst)
                for i in 1:12
                    dts[i] = dts[i + 1]
                end
                dts[13] = tmp
                return nothing
            end
            f(k4, u, p, t + dt)
            OrdinaryDiffEqCore.increment_nf!(integrator.stats, 1)
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
                    integrator.opts.reltol, integrator.opts.internalnorm,
                    t
                )
                calculate_residuals!(
                    atmpm1, utildem1, uprev, u, integrator.opts.abstol,
                    integrator.opts.reltol, integrator.opts.internalnorm,
                    t
                )
                calculate_residuals!(
                    atmpp1, utildep1, uprev, u, integrator.opts.abstol,
                    integrator.opts.reltol, integrator.opts.internalnorm,
                    t
                )
                errm2 = integrator.opts.internalnorm(atmpm2, t)
                errm1 = integrator.opts.internalnorm(atmpm1, t)
                errp1 = integrator.opts.internalnorm(atmpp1, t)
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
