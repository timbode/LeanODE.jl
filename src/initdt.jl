# Copied from OrdinaryDiffEqCore 3.33.1 (src/initdt.jl, `_ode_initdt_iip`), MIT
# licence; see LICENSE.md. Branches for DAEs, mass matrices, noise and
# non-`Array` states are deleted; the arithmetic is unchanged.
#
# The algorithm is Hairer, Nørsett and Wanner, Solving ODEs I, Sec. II.4,
# "Automatic step size control: starting step size". In its notation:
#   sk  = sc_i = atol + |u0_i| rtol        (the error scale per component)
#   d₀  = ‖u0 / sk‖,  d₁ = ‖f(t0, u0) / sk‖
#   dt₀ = h0 = 0.01 d₀/d₁                  (first guess)
#   d₂  = ‖(f(t0 + h0, u0 + h0 f0) - f0) / sk‖ / h0
#   dt₁ = h1 = (0.01 / max(d₁, d₂))^(1/order)
# and the result is min(100 h0, h1), bounded by dtmin and dtmax.

"""
    ode_initdt(integrator)

Return the initial step size, signed with the time direction, for the state
in `integrator`. Evaluates `f` twice: at `t0` into `integrator.fsallast`, and
once at `t0 + dt₀`. The array `integrator.cache.tmp` is used as scratch space
for the error scale, as upstream does.
"""
function ode_initdt(integrator)
    return _ode_initdt_iip(
        integrator.u, integrator.t, integrator.tdir, integrator.opts.dtmax,
        integrator.opts.abstol, integrator.opts.reltol, ode_default_norm,
        integrator.f, integrator.p,
        get_current_alg_order(integrator.alg, integrator.cache), integrator
    )
end

"""
    _ode_initdt_iip(u0, t, tdir, dtmax, abstol, reltol, internalnorm, f, p, order, integrator)

Hairer's starting-step algorithm for an in-place right-hand side `f`, as in
OrdinaryDiffEqCore's function of the same name. `order` is the order of the
method. Returns the step size multiplied by `tdir`.
"""
@muladd function _ode_initdt_iip(
        u0, t, tdir, dtmax, abstol, reltol, internalnorm,
        f, p, order, integrator
    )
    _tType = eltype(t)
    oneunit_tType = oneunit(_tType)
    dtmax_tdir = tdir * dtmax

    dtmin = nextfloat(max(integrator.opts.dtmin, eps(t)))
    smalldt = max(dtmin, convert(_tType, oneunit_tType * 1 // 10^(6)))

    sk = integrator.cache.tmp
    @inbounds @simd ivdep for i in eachindex(u0)
        sk[i] = abstol + internalnorm(u0[i], t) * reltol
    end

    f₀ = integrator.fsallast
    f(f₀, u0, p, t)

    if length(u0) > 0
        T = typeof(u0[1] / sk[1])
    else
        T = promote_type(eltype(u0), eltype(sk))
    end
    tmp = similar(u0, T)
    @inbounds @simd ivdep for i in eachindex(u0)
        tmp[i] = u0[i] / sk[i]
    end
    d₀ = internalnorm(tmp, t)

    @inbounds @simd ivdep for i in eachindex(u0)
        tmp[i] = f₀[i] / sk[i] * oneunit_tType
    end
    d₁ = internalnorm(tmp, t)

    if isnan(d₁)
        return tdir * dtmin
    end

    dt₀ = ifelse(
        (d₀ < 1 // 10^(5)) |
            (d₁ < 1 // 10^(5)), smalldt,
        convert(
            _tType,
            oneunit_tType * (
                (d₀ / d₁) /
                    100
            )
        )
    )
    dt₀ = min(dt₀, dtmax_tdir)

    if typeof(one(_tType)) <: AbstractFloat && dt₀ < 10eps(_tType) * oneunit(_tType)
        return tdir * max(smalldt, dtmin)
    end

    dt₀_tdir = tdir * dt₀

    u₁ = zero(u0)
    @inbounds @simd ivdep for i in eachindex(u0)
        u₁[i] = u0[i] + dt₀_tdir * f₀[i]
    end
    f₁ = zero(f₀)
    f(f₁, u₁, p, t + dt₀_tdir)

    length(u0) > 0 && f₀ == f₁ && return tdir * max(dtmin, 100dt₀)

    @inbounds @simd ivdep for i in eachindex(u0)
        tmp[i] = (f₁[i] - f₀[i]) / sk[i] * oneunit_tType
    end
    d₂ = internalnorm(tmp, t) / dt₀ * oneunit_tType

    max_d₁d₂ = max(d₁, d₂)
    if max_d₁d₂ <= 1 // Int64(10)^(15)
        dt₁ = max(convert(_tType, oneunit_tType * 1 // 10^(6)), dt₀ * 1 // 10^(3))
    else
        dt₁ = convert(
            _tType,
            oneunit_tType *
                10.0^(-(2 + log10(max_d₁d₂)) / order)
        )
    end
    return tdir * max(dtmin, min(100dt₀, dt₁, dtmax_tdir))
end
