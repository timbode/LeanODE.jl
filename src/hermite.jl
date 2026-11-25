# Cubic Hermite interpolation within a step (Hairer, Nørsett and Wanner,
# Solving ODEs I, p. 190): the cubic through the states y₀ and y₁ at both ends
# of a step of size dt, with the derivatives k₀ = f(y₀) and k₁ = f(y₁). With
# Δ = y₁ - y₀ and Θ ∈ [0, 1] the position within the step,
#
#   y(Θ)     = y₀ + Θ²(3 - 2Θ) Δ + Θ(Θ - 1)² dt k₀ + Θ²(Θ - 1) dt k₁
#   dt y′(Θ) =      6Θ(1 - Θ) Δ + (3Θ - 1)(Θ - 1) dt k₀ + Θ(3Θ - 2) dt k₁.
#
# OrdinaryDiffEq interpolates with this cubic for every method that has no
# interpolant of its own, and its RK4 estimates the error from the cubic.

"""
    hermite_weights(Θ, dt)

The weights `(wΔ, w₀, w₁)` of `Δ`, `k₀` and `k₁` in the cubic Hermite
interpolant at `Θ`: `y(Θ) = y₀ + wΔ Δ + w₀ k₀ + w₁ k₁`.
"""
hermite_weights(Θ, dt) = (Θ^2 * (3 - 2Θ), Θ * (Θ - 1)^2 * dt, Θ^2 * (Θ - 1) * dt)

"""
    hermite_slope_weights(Θ, dt)

The weights `(wΔ, w₀, w₁)` of `Δ`, `k₀` and `k₁` in the derivative of the
cubic Hermite interpolant at `Θ`, times the step size:
`dt y′(Θ) = wΔ Δ + w₀ k₀ + w₁ k₁`.
"""
hermite_slope_weights(Θ, dt) = (6Θ * (1 - Θ), (3Θ - 1) * (Θ - 1) * dt, Θ * (3Θ - 2) * dt)

"""
    hermite_sum(start, weights, Δ, k₀, k₁)

`start + wΔ Δ + w₀ k₀ + w₁ k₁` for one component, with
`weights = (wΔ, w₀, w₁)`.
"""
@inline function hermite_sum(start, (wΔ, w₀, w₁), Δ, k₀, k₁)
    return muladd(w₁, k₁, muladd(w₀, k₀, muladd(wΔ, Δ, start)))
end

"""
    hermite!(out, Θ, dt, y₀, y₁, k₀, k₁)

Write the cubic Hermite interpolant at `Θ` into `out`. All arrays have the
same shape, and `out` is none of the others. Returns `out`.
"""
function hermite!(out, Θ, dt, y₀, y₁, k₀, k₁)
    weights = hermite_weights(Θ, dt)
    @inbounds @simd ivdep for i in eachindex(out)
        out[i] = hermite_sum(y₀[i], weights, y₁[i] - y₀[i], k₀[i], k₁[i])
    end
    return out
end

"""
    hermite_pair!(out₁, out₂, Θ₁, Θ₂, dt, y₀, y₁, k₀, k₁)

Write the cubic Hermite interpolant at `Θ₁` into `out₁` and at `Θ₂` into
`out₂`, in one pass. All arrays have the same shape, and the outputs are none
of the inputs.
"""
function hermite_pair!(out₁, out₂, Θ₁, Θ₂, dt, y₀, y₁, k₀, k₁)
    weights₁ = hermite_weights(Θ₁, dt)
    weights₂ = hermite_weights(Θ₂, dt)
    @inbounds @simd ivdep for i in eachindex(out₁)
        Δ = y₁[i] - y₀[i]
        out₁[i] = hermite_sum(y₀[i], weights₁, Δ, k₀[i], k₁[i])
        out₂[i] = hermite_sum(y₀[i], weights₂, Δ, k₀[i], k₁[i])
    end
    return nothing
end

"""
    hermite_residual_norms(Θ₁, Θ₂, dt, y₀, y₁, k₀, k₁, f₁, f₂, abstol, reltol)

The scaled norms of the residual `dt (f(y(Θ)) - y′(Θ))` of the cubic Hermite
interpolant at `Θ₁` and at `Θ₂`, where `f₁` and `f₂` hold `f` at the
interpolated states. A residual of zero would mean that the cubic solves the
differential equation at that point. The scaling is that of `error_norm`.
Returns the two norms; one pass, no temporary array.
"""
function hermite_residual_norms(Θ₁, Θ₂, dt, y₀, y₁, k₀, k₁, f₁, f₂, abstol, reltol)
    weights₁ = hermite_slope_weights(Θ₁, dt)
    weights₂ = hermite_slope_weights(Θ₂, dt)
    total₁ = total₂ = 0.0
    @inbounds @simd for i in eachindex(y₁)
        Δ = y₁[i] - y₀[i]
        residual₁ = dt * f₁[i] - hermite_sum(zero(Δ), weights₁, Δ, k₀[i], k₁[i])
        residual₂ = dt * f₂[i] - hermite_sum(zero(Δ), weights₂, Δ, k₀[i], k₁[i])
        total₁ += scaled_square(residual₁, y₀[i], y₁[i], abstol, reltol)
        total₂ += scaled_square(residual₂, y₀[i], y₁[i], abstol, reltol)
    end
    return root_mean(total₁, y₁), root_mean(total₂, y₁)
end

"""
    HermiteCache

The caches of the methods without an interpolant of their own. For these,
OrdinaryDiffEq interpolates between steps with the cubic Hermite polynomial.
"""
const HermiteCache = Union{RK4Cache, BS3Cache, AdamsCache}

"""
    interpolant!(out, Θ, dt, y₀, y₁, k, cache::HermiteCache)

Write into `out` the state at `tprev + Θ dt` within the last step, by cubic
Hermite interpolation through `y₀` and `y₁` with the derivatives
`k[1] = f(y₀)` and `k[2] = f(y₁)`. Returns `out`.
"""
function interpolant!(out::Array, Θ, dt, y₀, y₁, k, cache::HermiteCache)
    return hermite!(out, Θ, dt, y₀, y₁, k[1], k[2])
end
