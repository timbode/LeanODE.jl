# Copied from OrdinaryDiffEqCore 3.33.1 (src/dense/generic_dense.jl,
# `hermite_interpolant!` for `Array`), MIT licence; see LICENSE.md.

"""
    HermiteCache

The caches of the methods without an interpolant of their own. For these,
OrdinaryDiffEq interpolates between steps with the cubic Hermite polynomial.
"""
const HermiteCache = Union{RK4Cache, BS3Cache, VCABM3Cache, VCABM4Cache, VCABM5Cache, VCABMCache}

"""
    _dv(differential_vars, i)

Whether component `i` is differential; always `true` here (no algebraic
components), kept so that the interpolant reads as upstream.
"""
@inline _dv(dv::Bool, _) = dv

"""
    interpolant!(out, Θ, dt, y₀, y₁, k, cache::HermiteCache)

Cubic Hermite interpolation (Hairer, Nørsett and Wanner, Solving ODEs I,
p. 190) through `y₀`, `y₁` with derivatives `k[1] = f(y₀)`, `k[2] = f(y₁)`,
at `tprev + Θ dt`. Returns `out`.
"""
@muladd function interpolant!(
        out::Array, Θ, dt, y₀, y₁, k, cache::HermiteCache
    )
    differential_vars = true
    @inbounds @simd ivdep for i in eachindex(out)
        out[i] = (1 - Θ) * y₀[i] + Θ * y₁[i] +
            _dv(differential_vars, i) * Θ * (Θ - 1) *
            ((1 - 2Θ) * (y₁[i] - y₀[i]) + (Θ - 1) * dt * k[1][i] + Θ * dt * k[2][i])
    end
    return out
end
