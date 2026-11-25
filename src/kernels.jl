# The loops over the state that the methods are built from. Each is one pass
# over arrays of the same shape, inlined into the method's step function.
#
# OrdinaryDiffEq writes these loops as broadcasts and computes the error
# estimate in three passes with two temporary arrays. The loops here do the
# same arithmetic in a different order of operations and in fewer passes, so
# results agree with OrdinaryDiffEq to rounding error, not bit for bit.
#
# None of them uses `@fastmath`: a state that became infinite or NaN must stay
# so, because the solve detects instability from it.

"""
    weighted_sum(start, i, weight₁, k₁, weight₂, k₂, …)

`start + weight₁ k₁[i] + weight₂ k₂[i] + …`, with each product fused into
the running sum (`muladd`). The weights are numbers and the `k`s arrays.
"""
@inline weighted_sum(sum, i) = sum
Base.@propagate_inbounds function weighted_sum(sum, i, weight, k, rest...)
    return weighted_sum(muladd(weight, k[i], sum), i, rest...)
end

"""
    combine!(out, start, weight₁, k₁, weight₂, k₂, …)

Write `start + weight₁ k₁ + weight₂ k₂ + …` into `out`, component by
component. All arrays must have the same shape, and `out` must be none of
the others.
"""
@inline function combine!(out, start, terms::Vararg{Any, N}) where {N}
    @inbounds @simd ivdep for i in eachindex(out)
        out[i] = weighted_sum(start[i], i, terms...)
    end
    return nothing
end

"""
    magnitude(x)

`|x|` without the protection against overflow that `abs` of a complex number
has; OrdinaryDiffEq's error norm takes it the same way.
"""
@inline magnitude(x::Float64) = abs(x)
@inline magnitude(z::ComplexF64) = sqrt(abs2(z))

"""
    scaled_square(err, u₀, u₁, abstol, reltol)

`|err / (abstol + max(|u₀|, |u₁|) reltol)|²`: the squared error of one
component, relative to the tolerances.
"""
@inline function scaled_square(err, u₀, u₁, abstol, reltol)
    m₀ = magnitude(u₀)
    m₁ = magnitude(u₁)
    return abs2(err / muladd(ifelse(m₁ > m₀, m₁, m₀), reltol, abstol))
end

"""
    inverse_scale(u₀, u₁, abstol, reltol)

`1 / (abstol + max(|u₀|, |u₁|) reltol)`: the factor that scales an error of
one component relative to the tolerances. Several errors of the same
component then share one division.
"""
@inline function inverse_scale(u₀, u₁, abstol, reltol)
    m₀ = magnitude(u₀)
    m₁ = magnitude(u₁)
    return inv(muladd(ifelse(m₁ > m₀, m₁, m₀), reltol, abstol))
end

"""
    root_mean(total, u)

`sqrt(total / n)` for the sum `total` over the `n` components of `u`; zero
for an empty `u`.
"""
@inline root_mean(total, u) = sqrt(total / max(length(u), 1))

"""
    error_norm(uprev, u, abstol, reltol, weight₁, k₁, weight₂, k₂, …)

The scaled error estimate of a step from `uprev` to `u`: the root mean square
over the components of `errᵢ / (abstol + max(|uprevᵢ|, |uᵢ|) reltol)`, where
`err = weight₁ k₁ + weight₂ k₂ + …`. A step is acceptable if the result is at
most 1. One pass, no temporary array.
"""
@inline function error_norm(uprev, u, abstol, reltol, terms::Vararg{Any, N}) where {N}
    total = 0.0
    @inbounds @simd for i in eachindex(u)
        err = weighted_sum(zero(eltype(u)), i, terms...)
        total += scaled_square(err, uprev[i], u[i], abstol, reltol)
    end
    return root_mean(total, u)
end
