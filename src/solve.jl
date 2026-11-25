# Copied from OrdinaryDiffEqCore 3.33.1 (src/solve.jl: `_ode_init`, `solve!`,
# `handle_dt!`, `initialize_saveat`; src/integrators/integrator_utils.jl: the
# step loop) and SciMLBase 2.155.2 (src/integrator_interface.jl: `check_error`),
# MIT licence; see LICENSE.md. Branches for callbacks, DAEs, noise, composite
# methods, units, discontinuities and user tstops are deleted. Each method of a
# function keeps its upstream name, so the two can be read side by side.

# ---------------------------------------------------------------- setup

"""
    default_tolerance(u, scale)

OrdinaryDiffEq's default tolerance for the element type of `u`: `scale` (a
`Rational`) converted to that type, then its real part.
"""
function default_tolerance(u, scale)
    uBottomEltype = eltype(u)
    return real(convert(uBottomEltype, oneunit(uBottomEltype) * scale))
end

"""
    initialize_saveat(saveat, tspan)

The times to save at strictly after `tspan[1]` and up to `tspan[2]`,
multiplied by the time direction and sorted ascending. `saveat` is a
collection of times, or a number meaning every `saveat` time units.
"""
function initialize_saveat(saveat, tspan)
    saveat_internal = Float64[]
    t0, tf = tspan
    tdir = sign(tf - t0)
    tdir_t0 = tdir * t0
    tdir_tf = tdir * tf
    if saveat isa Number
        directional_saveat = tdir * abs(saveat)
        for t in (t0 + directional_saveat):directional_saveat:tf
            push!(saveat_internal, tdir * t)
        end
    elseif !isempty(saveat)
        for t in saveat
            tdir_t = tdir * t
            tdir_t0 < tdir_t ≤ tdir_tf && push!(saveat_internal, tdir_t)
        end
    end
    return sort!(saveat_internal)
end

"""
    build_options(prob, alg, settings)

The `Options` of a solve, from the keyword arguments of `solve` collected in
the named tuple `settings`. Unset tolerances and controller parameters take
OrdinaryDiffEq's defaults for an adaptive method of order `alg_order(alg)`.
"""
function build_options(prob, alg, settings)
    (; saveat, save_everystep, save_start, save_end, dtmin, dtmax, adaptive,
        abstol, reltol, maxiters) = settings
    tspan = prob.tspan
    tdir = sign(tspan[end] - tspan[1])
    u = prob.u0

    for (name, tolerance) in (("abstol", abstol), ("reltol", reltol))
        tolerance === nothing || tolerance isa Number || throw(
            ArgumentError("$name must be a number, got a $(typeof(tolerance)); a tolerance per component is not supported")
        )
    end
    abstol_internal = abstol === nothing ? default_tolerance(u, 1 // 10^6) : real.(abstol)
    reltol_internal = reltol === nothing ? default_tolerance(u, 1 // 10^3) : real.(reltol)
    dtmax > zero(dtmax) && tdir < 0 && (dtmax *= tdir) # Allow positive dtmax, but auto-convert

    QT = Float64
    order = alg_order(alg)
    save_end_user = save_end
    save_end = save_end === nothing ?
        save_everystep || isempty(saveat) || saveat isa Number ||
        tspan[2] in saveat : save_end

    return Options(
        maxiters, save_everystep, adaptive, abstol_internal, reltol_internal,
        convert(QT, 9 // 10),       # gamma
        convert(QT, 10),            # qmax
        convert(QT, 1 // 5),        # qmin
        convert(QT, 1),             # qsteady_max
        convert(QT, 1),             # qsteady_min
        convert(QT, 1 // 10^4),     # qoldinit
        convert(QT, pi_beta1(order)), convert(QT, pi_beta2(order)),
        Float64(dtmax), Float64(dtmin),
        [tdir * tspan[2]], initialize_saveat(saveat, tspan), saveat,
        save_start, save_end, save_end_user
    )
end

"""
    init(prob, alg, settings)

Set up the integrator: copy the initial state, build the method's cache, save
the initial state if asked, evaluate `f(u0)` and choose the initial step size.
"""
function init(prob::ODEProblem, alg::LeanAlgorithm, settings)
    (; dt, adaptive) = settings
    if !adaptive && isnothing(dt)
        throw(ArgumentError("Fixed timestep methods require a choice of dt or choosing the tstops"))
    end
    opts = build_options(prob, alg, settings)
    tspan = prob.tspan
    tdir = sign(tspan[end] - tspan[1])
    t = tspan[1]
    u = copy(prob.u0)
    uprev = copy(u)
    rate_prototype = copy(u)
    cache = alg_cache(alg, u, rate_prototype, uprev)
    fsalfirst, fsallast = get_fsalfirstlast(cache)
    _dt = isnothing(dt) ? zero(t) : Float64(dt)

    integrator = Integrator(
        prob.f, prob.p, alg, cache, opts, Stats(), tdir,
        u, uprev, typeof(u)[], fsalfirst, fsallast,
        t, t, tspan[2], _dt, _dt, _dt,
        1.0,                # EEst = oneunit(EEstT)
        opts.qoldinit,      # qold
        1.0,                # q11
        0, 0, 0,            # iter, success_iter, saveiter
        false, false, zero(t), true, false,
        ReturnCode.Default, Float64[], typeof(u)[]
    )
    opts.save_start && save_current!(integrator)
    initialize!(integrator, cache)
    handle_dt!(integrator, dt)
    return integrator
end

"""
    handle_dt!(integrator, dt)

Set the initial step size: from Hairer's algorithm (`ode_initdt`) if the
caller gave none and the method is adaptive, otherwise the caller's `dt`
with the sign of the time direction.
"""
function handle_dt!(integrator, dt)
    if isnothing(dt) && iszero(integrator.dt) && integrator.opts.adaptive
        integrator.dt = ode_initdt(integrator)
        integrator.dtpropose = integrator.dt
        integrator.stats.nf += 2
        if sign(integrator.dt) != integrator.tdir && !iszero(integrator.dt) &&
                !isnan(integrator.dt)
            error("Automatic dt setting has the wrong sign. Exiting. Please report this error.")
        end
    elseif integrator.opts.adaptive && integrator.dt > zero(integrator.dt) &&
            integrator.tdir < 0
        integrator.dt *= integrator.tdir # Allow positive dt, but auto-convert
    end
    return nothing
end

# ---------------------------------------------------------------- one step

"""
    loopheader!(integrator)

Prepare the next step. After an accepted step, move the state forward
(`apply_step!`); after a rejected one, shrink the step size. Then bound the
step size by `dtmin`, `dtmax` and the distance to the next stop time.
"""
function loopheader!(integrator)
    if integrator.iter > 0
        if !integrator.opts.adaptive || integrator.accept_step
            integrator.success_iter += 1
            apply_step!(integrator)
        elseif integrator.opts.adaptive && !integrator.accept_step
            step_reject_controller!(integrator)
        end
    end
    integrator.iter += 1
    fix_dt_at_bounds!(integrator)
    modify_dt_for_tstops!(integrator)
    return nothing
end

"""
    apply_step!(integrator)

Make the accepted step the starting point of the next: `u` becomes `uprev`
and `f(u)` becomes `f(uprev)` (first same as last), and the proposed step
size is taken.

OrdinaryDiffEq copies `u` into `uprev` and `fsallast` into `fsalfirst`.
LeanODE exchanges the arrays instead, which saves two passes over the state
per step and changes no value that is read afterwards: the next step
overwrites `u` and `fsallast` before reading them. Until then
`state_in_uprev` marks `u` as outdated.
"""
function apply_step!(integrator)
    exchange_state!(integrator)
    integrator.dt = integrator.dtpropose
    modify_dt_for_tstops!(integrator)
    return nothing
end

"""
    exchange_state!(integrator)

Exchange `u` with `uprev` and `fsallast` with `fsalfirst`, unless that was
already done since the last step. The second case arises when a step is
accepted without being taken (see `handle_tstop_step!`).
"""
function exchange_state!(integrator)
    integrator.state_in_uprev && return nothing
    integrator.u, integrator.uprev = integrator.uprev, integrator.u
    exchange_fsal!(integrator, integrator.cache)
    integrator.state_in_uprev = true
    return nothing
end

"""
    perform_step!(integrator)

Take one step with the integrator's method. Afterwards `u` holds the result.
"""
function perform_step!(integrator)
    perform_step!(integrator, integrator.cache)
    integrator.state_in_uprev = false
    return nothing
end

"""
    fix_dt_at_bounds!(integrator)

Clamp the step size to `[timedepentdtmin, dtmax]` in the time direction.
"""
function fix_dt_at_bounds!(integrator)
    if integrator.tdir > 0
        integrator.dt = min(integrator.opts.dtmax, integrator.dt)
    else
        integrator.dt = max(integrator.opts.dtmax, integrator.dt)
    end
    dtmin = timedepentdtmin(integrator.t, integrator.opts.dtmin)
    if integrator.tdir > 0
        integrator.dt = max(integrator.dt, dtmin)
    else
        integrator.dt = min(integrator.dt, dtmin)
    end
    return nothing
end

"""
    modify_dt_for_tstops!(integrator)

Shorten the step so that it does not pass the next stop time. If the step
reaches the stop time (within 100 ulp), flag it, so that the step ends
exactly on it. Keeps the unshortened step size in `dtpropose`.
"""
function modify_dt_for_tstops!(integrator)
    tstops = integrator.opts.tstops
    if !isempty(tstops)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = first(tstops)
        distance_to_tstop = abs(tdir_tstop - tdir_t)
        tstop_tol = if integrator.t isa AbstractFloat && isfinite(tdir_tstop) &&
                isfinite(integrator.t)
            100 * eps(
                float(
                    max(abs(integrator.t), abs(tdir_tstop)) /
                        oneunit(integrator.t)
                )
            ) * oneunit(integrator.t)
        else
            zero(distance_to_tstop)
        end

        if integrator.opts.adaptive
            original_dt = abs(integrator.dt)
            integrator.dtpropose = integrator.tdir * original_dt
            if original_dt + tstop_tol < distance_to_tstop
                set_tstop_flag!(integrator, false)
            else
                set_tstop_flag!(integrator, true, integrator.tdir * tdir_tstop)
            end
            integrator.dt = integrator.tdir * min(original_dt, distance_to_tstop)
        elseif iszero(integrator.dtcache)
            integrator.dt = integrator.tdir * distance_to_tstop
            set_tstop_flag!(integrator, true, integrator.tdir * tdir_tstop)
        else
            if abs(integrator.dtcache) + tstop_tol < distance_to_tstop
                set_tstop_flag!(integrator, false)
            else
                set_tstop_flag!(integrator, true, integrator.tdir * tdir_tstop)
            end
            integrator.dt = integrator.tdir *
                min(abs(integrator.dtcache), distance_to_tstop)
        end
    else
        set_tstop_flag!(integrator, false)
    end
    return nothing
end

"""
    set_tstop_flag!(integrator, is_tstop, target = nothing)

Record whether the next step ends on a stop time, and which one.
"""
function set_tstop_flag!(integrator, is_tstop::Bool, target = nothing)
    integrator.next_step_tstop = is_tstop
    if is_tstop && target !== nothing
        integrator.tstop_target = target
    end
    return nothing
end

"""
    handle_tstop_step!(integrator)

Take a step that ends on a stop time. A step shorter than the floating-point
resolution of `t` is not taken but counted as accepted.
"""
function handle_tstop_step!(integrator)
    if integrator.t isa AbstractFloat && abs(integrator.dt) < eps(abs(integrator.t))
        integrator.accept_step = true
    else
        perform_step!(integrator)
    end
    return nothing
end

"""
    loopfooter!(integrator)

Decide on the step just taken. An accepted step advances `t` (to the stop
time exactly, if flagged), proposes the next step size and saves; a rejected
step only counts.
"""
function loopfooter!(integrator)
    integrator.do_error_check = true
    ttmp = integrator.t + integrator.dt
    if integrator.opts.adaptive
        q = stepsize_controller!(integrator)
        integrator.accept_step = accept_step_controller(integrator)
        if integrator.accept_step # Accept
            integrator.stats.naccept += 1
            integrator.tprev = integrator.t
            if integrator.next_step_tstop
                integrator.dt = integrator.dtpropose
            end
            integrator.t = fixed_t_for_tstop_error!(integrator, ttmp)
            dtnew = step_accept_controller!(integrator, q) * oneunit(integrator.dt)
            calc_dt_propose!(integrator, dtnew)
            savevalues!(integrator) # what `handle_callbacks!` does without callbacks
        else # Reject
            integrator.stats.nreject += 1
        end
    else # Not adaptive
        integrator.stats.naccept += 1
        integrator.tprev = integrator.t
        integrator.t = fixed_t_for_tstop_error!(integrator, ttmp)
        integrator.accept_step = true
        integrator.dtpropose = integrator.dt
        savevalues!(integrator)
    end
    return nothing
end

"""
    fixed_t_for_tstop_error!(integrator, ttmp)

The time at the end of the accepted step: the stop time if the step was
flagged to end on one (clearing the flag), else `ttmp = t + dt`.
"""
function fixed_t_for_tstop_error!(integrator, ttmp)
    if integrator.next_step_tstop
        set_tstop_flag!(integrator, false)
        return integrator.tstop_target
    else
        return ttmp
    end
end

"""
    calc_dt_propose!(integrator, dtnew)

Store `dtnew`, clamped to `[timedepentdtmin, dtmax]`, as the proposed next
step size.
"""
function calc_dt_propose!(integrator, dtnew)
    dtpropose = integrator.tdir * min(abs(integrator.opts.dtmax), abs(dtnew))
    dtpropose = integrator.tdir *
        max(abs(dtpropose), timedepentdtmin(integrator.t, integrator.opts.dtmin))
    integrator.dtpropose = dtpropose
    return nothing
end

"""
    handle_tstop!(integrator)

Remove the stop time just reached (and any copies of it) from the list.
"""
function handle_tstop!(integrator)
    tstops = integrator.opts.tstops
    if !isempty(tstops)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = first(tstops)
        if tdir_t == tdir_tstop
            while tdir_t == tdir_tstop # remove all redundant copies
                popfirst!(tstops)
                !isempty(tstops) ? (tdir_tstop = first(tstops)) : break
            end
        elseif tdir_t > tdir_tstop
            error("Something went wrong. Integrator stepped past tstops but the algorithm was dtchangeable. Please report this error.")
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the loop

"""
    check_error(integrator)

The return code the solve would end with now: `ReturnCode.Success` if it may
continue, else the reason to stop (NaN step size, too many steps, step size
at `dtmin` or below floating-point resolution, infinite or NaN state).
"""
function check_error(integrator)
    if integrator.retcode ∉ (ReturnCode.Success, ReturnCode.Default)
        return integrator.retcode
    end
    opts = integrator.opts
    if isnan(integrator.dt)
        return ReturnCode.DtNaN
    end
    if integrator.iter > opts.maxiters
        return ReturnCode.MaxIters
    end
    step_accepted = integrator.accept_step
    if opts.adaptive
        if abs(integrator.dt) <= abs(opts.dtmin) &&
                (
                !step_accepted ||
                    integrator.t + integrator.dt < integrator.tdir * first(opts.tstops)
            )
            return ReturnCode.DtLessThanMin
        elseif !step_accepted && integrator.t isa AbstractFloat &&
                abs(integrator.dt) <= abs(eps(integrator.t))
            return ReturnCode.Unstable
        end
    end
    if step_accepted && infinite_or_giant(latest_state(integrator))
        return ReturnCode.Unstable
    end
    return ReturnCode.Success
end

"""
    check_error!(integrator)

Like `check_error`, but also store the code; if it is not `Success`, finish
the solution with `postamble!`.
"""
function check_error!(integrator)
    code = check_error(integrator)
    integrator.retcode = code
    if code != ReturnCode.Success
        postamble!(integrator)
    end
    return code
end

"""
    solve!(integrator)

Step until the end of the time span or until `check_error` stops the solve.
Returns the integrator; the solution is in `integrator.ts` and `integrator.us`.
"""
function solve!(integrator)
    tstops = integrator.opts.tstops
    @inbounds while !isempty(tstops)
        first_tstop = first(tstops)
        while integrator.tdir * integrator.t < first_tstop
            loopheader!(integrator)
            if integrator.do_error_check && check_error!(integrator) != ReturnCode.Success
                return integrator
            end
            if integrator.next_step_tstop
                handle_tstop_step!(integrator)
            else
                perform_step!(integrator)
            end
            should_exit = integrator.next_step_tstop
            loopfooter!(integrator)
            if isempty(tstops) || should_exit
                break
            end
        end
        handle_tstop!(integrator)
    end
    postamble!(integrator)
    if integrator.retcode == ReturnCode.Default
        integrator.retcode = ReturnCode.Success
    end
    return integrator
end

"""
    solve(prob, alg; kwargs...)

Integrate `prob` with the method `alg` (`Tsit5()` or `RK4()`) and return an
[`ODESolution`](@ref). The keywords and their defaults are OrdinaryDiffEq's:

- `abstol = 1e-6`, `reltol = 1e-3`: absolute and relative tolerance, one
  number each for all components.
- `saveat = ()`: times to save at, or a number meaning every `saveat` time
  units. States between steps come from the method's interpolant.
- `save_everystep = isempty(saveat)`: also save the state after every step.
- `save_start`, `save_end`: save the initial and the final state. By default
  they are saved unless `saveat` is given and leaves them out.
- `dt`: initial step size; by default chosen automatically. Required with
  `adaptive = false`, where it is the fixed step size.
- `adaptive = true`: control the step size by the error estimate.
- `dtmin = 0`, `dtmax = tspan[2] - tspan[1]`: bounds on the step size.
- `maxiters = 10^6`: maximum number of attempted steps.

The solve stops early with a return code other than `ReturnCode.Success`
when the step size becomes NaN or too small, the state becomes infinite or
NaN, or `maxiters` is exceeded.

# Throws
- `ArgumentError` if `adaptive = false` and no `dt` is given, or if a
  tolerance is not a number.
"""
function solve(
        prob::ODEProblem, alg::LeanAlgorithm;
        saveat = (),
        save_everystep = isempty(saveat),
        save_start = save_everystep || isempty(saveat) ||
            saveat isa Number || prob.tspan[1] in saveat,
        save_end = nothing,
        dt = nothing,
        dtmin = eltype(prob.tspan)(0),
        dtmax = eltype(prob.tspan)((prob.tspan[end] - prob.tspan[1])),
        adaptive = true,
        abstol = nothing,
        reltol = nothing,
        maxiters = 1000000
    )
    settings = (;
        saveat, save_everystep, save_start, save_end, dt, dtmin, dtmax,
        adaptive, abstol, reltol, maxiters,
    )
    integrator = init(prob, alg, settings)
    solve!(integrator)
    return ODESolution(
        integrator.ts, integrator.us, integrator.retcode, integrator.stats, prob, alg
    )
end
