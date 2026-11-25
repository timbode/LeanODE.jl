# Copied from OrdinaryDiffEqCore 3.33.1 (src/integrators/integrator_utils.jl:
# `_savevalues!`, `solution_endpoint_match_cur_integrator!`, `_postamble!`;
# src/dense/generic_dense.jl: `ode_interpolant`) and RecursiveArrayTools 3.54.0
# (`copyat_or_push!`), MIT licence; see LICENSE.md. Branches for callbacks,
# dense output, `save_idxs`, noise and composite methods are deleted.

"""
    copyat_or_push!(a, i, x, perform_copy = true)

Store `x` as element `i` of `a`: overwrite it if `a` already has an element
`i`, else push. With `perform_copy`, an array `x` is stored as a copy; numbers
are stored as they are.
"""
function copyat_or_push!(a::AbstractVector, i::Int, x, perform_copy = true)
    @inbounds if length(a) >= i
        if !(eltype(a) <: AbstractArray) || !perform_copy
            a[i] = x
        elseif length(a[i]) == length(x)
            copyto!(a[i], x)
        else
            a[i] = copy(x)
        end
    else
        push!(a, perform_copy ? copy(x) : x)
    end
    return nothing
end

"""
    save_current!(integrator)

Save the current time and a copy of the current state as the next entry of
the solution.
"""
function save_current!(integrator)
    integrator.saveiter += 1
    copyat_or_push!(integrator.ts, integrator.saveiter, integrator.t)
    copyat_or_push!(integrator.us, integrator.saveiter, integrator.u)
    return nothing
end

"""
    interp_at_saveat(Θ, integrator)

State at `tprev + Θ dt`, from the interpolant of the current step. Returns a
new array.
"""
function interp_at_saveat(Θ, integrator)
    y₁ = integrator.u
    out = similar(y₁, eltype(first(y₁) * oneunit(Θ)))
    copyto!(out, y₁)
    return interpolant!(
        out, Θ, integrator.dt, integrator.uprev, y₁, integrator.k, integrator.cache
    )
end

"""
    savevalues!(integrator)

Save what the current step calls for: every `saveat` time it passed (by
interpolation, or the state itself when the time is hit exactly), and with
`save_everystep` the state at the end of the step.
"""
function savevalues!(integrator)
    tdir_t = integrator.tdir * integrator.t
    saveat = integrator.opts.saveat
    while !isempty(saveat) && first(saveat) <= tdir_t # Perform saveat
        integrator.saveiter += 1
        curt = integrator.tdir * popfirst!(saveat)
        if curt != integrator.t # If <t, interpolate
            Θ = (curt - integrator.tprev) / integrator.dt
            val = interp_at_saveat(Θ, integrator)
            copyat_or_push!(integrator.ts, integrator.saveiter, curt)
            copyat_or_push!(integrator.us, integrator.saveiter, val, false)
        else # ==t, just save
            if curt == integrator.tspan_end && !integrator.opts.save_end
                integrator.saveiter -= 1
                continue
            end
            copyat_or_push!(integrator.ts, integrator.saveiter, integrator.t)
            copyat_or_push!(integrator.us, integrator.saveiter, integrator.u)
        end
    end
    if integrator.opts.save_everystep &&
            (
            isempty(integrator.ts) ||
                (integrator.t !== integrator.ts[end] || iszero(integrator.dt)) &&
                (integrator.opts.save_end || integrator.t !== integrator.tspan_end)
        )
        save_current!(integrator)
    end
    return nothing
end

"""
    solution_endpoint_match_cur_integrator!(integrator)

Save the final state if `save_end` asks for it and it is not saved yet.
"""
function solution_endpoint_match_cur_integrator!(integrator)
    opts = integrator.opts
    if opts.save_end &&
            (
            integrator.saveiter == 0 ||
                integrator.ts[integrator.saveiter] != integrator.t &&
                (
                (opts.save_end_user isa Bool && opts.save_end_user) ||
                    integrator.t ∈ opts.saveat_cache ||
                    integrator.t == integrator.tspan_end ||
                    isempty(opts.saveat_cache)
            )
        )
        save_current!(integrator)
    end
    return nothing
end

"""
    postamble!(integrator)

Finish the solution: save the final state if needed and trim the saved
arrays to the number of saves.
"""
function postamble!(integrator)
    solution_endpoint_match_cur_integrator!(integrator)
    resize!(integrator.ts, integrator.saveiter)
    resize!(integrator.us, integrator.saveiter)
    return nothing
end
