# Copied from OrdinaryDiffEqCore 3.33.1 (src/integrators/controllers.jl, the
# legacy `PIController`, and `get_current_qmax`), MIT licence; see LICENSE.md.
#
# The PI controller sets the next step size h_new = h / q with
#   q = EEst^beta1 / qold^beta2 / gamma,   clamped to [1/qmax, 1/qmin],
# where EEst is the scaled error of the current step and qold that of the
# last accepted one (Hairer and Wanner, Solving ODEs II, Sec. IV.2). A step is
# accepted if EEst ≤ 1. Powers are taken with `fastpower`, an approximation
# through a Float32 logarithm, exactly as upstream.

"""
    pi_beta1(order)

Default exponent `beta1 = 7 / (10 order)` of the PI controller for a method of
the given order.
"""
pi_beta1(order) = 7 // (10order)

"""
    pi_beta2(order)

Default exponent `beta2 = 2 / (5 order)` of the PI controller for a method of
the given order.
"""
pi_beta2(order) = 2 // (5order)

"""
    get_current_qmax(integrator, qmax)

Largest allowed growth factor of the step size: `qmax`, except before the
first accepted step, where it is 10⁴ (as Sundials' CVODE does).
"""
@inline function get_current_qmax(integrator, qmax)
    if integrator.success_iter == 0
        return typeof(qmax)(10000)
    end
    return qmax
end

"""
    stepsize_controller!(integrator)

Return the factor `q` by which the current step size is divided to obtain the
next one, from the error estimate `integrator.EEst`. Stores `EEst^beta1` in
`integrator.q11` for use by a rejection.
"""
@inline function stepsize_controller!(integrator)
    (; qold) = integrator
    (; qmin, qmax, gamma) = integrator.opts
    qmax = get_current_qmax(integrator, qmax)
    (; beta1, beta2) = integrator.opts
    EEst = integrator.EEst

    if iszero(EEst)
        q = inv(qmax)
    else
        q11 = fastpower(EEst, convert(typeof(EEst), beta1))
        q = q11 / fastpower(qold, convert(typeof(EEst), beta2))
        integrator.q11 = q11
        @fastmath q = max(inv(qmax), min(inv(qmin), q / gamma))
    end
    return q
end

"""
    step_accept_controller!(integrator, q)

Return the next step size after an accepted step, `dt / q`; a factor `q`
inside `[qsteady_min, qsteady_max]` is replaced by 1. Updates the controller
memory `integrator.qold`.
"""
function step_accept_controller!(integrator, q)
    (; qsteady_min, qsteady_max, qoldinit) = integrator.opts
    EEst = integrator.EEst

    if qsteady_min <= q <= qsteady_max
        q = one(q)
    end
    integrator.qold = max(EEst, qoldinit)
    return integrator.dt / q # new dt
end

"""
    step_reject_controller!(integrator)

Shrink `integrator.dt` after a rejected step, by the factor
`min(1/qmin, q11/gamma)`.
"""
function step_reject_controller!(integrator)
    (; q11) = integrator
    (; qmin, gamma) = integrator.opts
    return integrator.dt /= min(inv(qmin), q11 / gamma)
end

"""
    accept_step_controller(integrator)

`true` if the current step is accepted, that is, if its scaled error estimate
is at most 1.
"""
@inline accept_step_controller(integrator) = integrator.EEst <= 1
