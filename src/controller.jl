# Copied from OrdinaryDiffEqCore 3.33.1 (src/integrators/controllers.jl, the
# legacy `PIController` and `IController`, and `get_current_qmax`;
# src/alg_utils.jl, the choice between them), MIT licence; see LICENSE.md.
#
# Both controllers set the next step size h_new = h / q from the scaled error
# EEst of the current step, with q clamped to [1/qmax, 1/qmin], and accept a
# step if EEst ≤ 1. Powers are taken with `fastpower`, an approximation
# through a Float32 logarithm, exactly as upstream.
#
# The PI controller (Hairer and Wanner, Solving ODEs II, Sec. IV.2) uses
#   q = EEst^beta1 / qold^beta2 / gamma,
# where qold is the error of the last accepted step. Every method uses this
# controller unless it says otherwise.
#
# The integral controller (Hairer, Nørsett and Wanner, Solving ODEs I,
# Sec. II.4) uses
#   q = EEst^(1 / (k + 1)) / gamma,
# where k is the order of the error estimate. The variable-order VCABM uses
# it, with its current order as k.

"""
    PIController

Selects the proportional–integral step-size controller.
"""
struct PIController end

"""
    IController

Selects the integral step-size controller.
"""
struct IController end

"""
    isstandard(alg)

Whether the method `alg` uses the integral controller instead of the PI
controller. (The name is OrdinaryDiffEq's.)
"""
isstandard(alg::LeanAlgorithm) = false

"""
    default_controller(alg)

The step-size controller of the method `alg`.
"""
@inline default_controller(alg) = isstandard(alg) ? IController() : PIController()

"""
    get_current_alg_order(alg, cache)

The order of the method `alg` in its current state `cache`; it sets the
initial step size.
"""
get_current_alg_order(alg, cache) = alg_order(alg)

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
next one, from the error estimate `integrator.EEst`, with the controller of
the integrator's method.
"""
@inline function stepsize_controller!(integrator)
    return stepsize_controller!(integrator, default_controller(integrator.alg))
end

"""
    step_accept_controller!(integrator, q)

Return the next step size after an accepted step, `dt / q`, with the
controller of the integrator's method.
"""
@inline function step_accept_controller!(integrator, q)
    return step_accept_controller!(integrator, default_controller(integrator.alg), q)
end

"""
    step_reject_controller!(integrator)

Shrink `integrator.dt` after a rejected step, with the controller of the
integrator's method.
"""
@inline function step_reject_controller!(integrator)
    step_reject_controller!(integrator, default_controller(integrator.alg))
    return nothing
end

"""
    stepsize_controller!(integrator, ::PIController)

The PI controller's factor `q`. Stores `EEst^beta1` in `integrator.q11` for
use by a rejection.
"""
@inline function stepsize_controller!(integrator, controller::PIController)
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
    step_accept_controller!(integrator, ::PIController, q)

Return the next step size after an accepted step, `dt / q`; a factor `q`
inside `[qsteady_min, qsteady_max]` is replaced by 1. Updates the controller
memory `integrator.qold`.
"""
function step_accept_controller!(integrator, controller::PIController, q)
    (; qsteady_min, qsteady_max, qoldinit) = integrator.opts
    EEst = integrator.EEst

    if qsteady_min <= q <= qsteady_max
        q = one(q)
    end
    integrator.qold = max(EEst, qoldinit)
    return integrator.dt / q # new dt
end

"""
    step_reject_controller!(integrator, ::PIController)

Shrink `integrator.dt` after a rejected step, by the factor
`min(1/qmin, q11/gamma)`.
"""
function step_reject_controller!(integrator, controller::PIController)
    (; q11) = integrator
    (; qmin, gamma) = integrator.opts
    return integrator.dt /= min(inv(qmin), q11 / gamma)
end

"""
    stepsize_controller!(integrator, ::IController)

The integral controller's factor `q`. Stores the step size a rejection
would continue with, `dt / q`, in `integrator.qold`.
"""
@inline function stepsize_controller!(integrator, controller::IController)
    (; qmin, qmax, gamma) = integrator.opts
    qmax = get_current_qmax(integrator, qmax)
    EEst = integrator.EEst

    if iszero(EEst)
        q = inv(qmax)
    else
        expo = 1 / (get_current_adaptive_order(integrator.alg, integrator.cache) + 1)
        qtmp = fastpower(EEst, expo) / gamma
        @fastmath q = max(inv(qmax), min(inv(qmin), qtmp))
        # TODO: Shouldn't this be in `step_accept_controller!` as for the PI controller?
        integrator.qold = integrator.dt / q
    end
    return q
end

"""
    step_accept_controller!(integrator, ::IController, q)

Return the next step size after an accepted step, `dt / q`; a factor `q`
inside `[qsteady_min, qsteady_max]` is replaced by 1.
"""
function step_accept_controller!(integrator, controller::IController, q)
    (; qsteady_min, qsteady_max) = integrator.opts

    if qsteady_min <= q <= qsteady_max
        q = one(q)
    end
    return integrator.dt / q # new dt
end

"""
    step_reject_controller!(integrator, ::IController)

After a rejected step, continue with the step size that
`stepsize_controller!` stored in `integrator.qold`.
"""
function step_reject_controller!(integrator, controller::IController)
    (; qold) = integrator
    return integrator.dt = qold
end

"""
    accept_step_controller(integrator)

`true` if the current step is accepted, that is, if its scaled error estimate
is at most 1.
"""
@inline accept_step_controller(integrator) = integrator.EEst <= 1
