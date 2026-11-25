# Copied from DiffEqBase 6.218.0 (src/common_defaults.jl, src/utils.jl), MIT
# licence; see LICENSE.md. Only the methods for `Array` and scalars are kept.

"""
    ode_default_norm(u, t)

Error norm of OrdinaryDiffEq (`ODE_DEFAULT_NORM`). For a scalar it is `abs(u)`.
For an array it is the root mean square `sqrt(Σ|uᵢ|² / n)`, accumulated in a
`@fastmath` loop exactly as upstream. `t` is unused; it is part of the
signature that OrdinaryDiffEq passes.
"""
ode_default_norm(u::Union{AbstractFloat, Complex}, t) = @fastmath abs(u)

function ode_default_norm(u::Array{T}, t) where {T <: Union{AbstractFloat, Complex}}
    x = zero(T)
    @inbounds @fastmath for ui in u
        x += abs2(ui)
    end
    return Base.FastMath.sqrt_fast(real(x) / max(length(u), 1))
end

"""
    timedepentdtmin(t, dtmin)

Smallest step size allowed at time `t`: `dtmin`, but at least `eps(t)`.
(The spelling follows DiffEqBase.)
"""
timedepentdtmin(t::AbstractFloat, dtmin) = abs(max(eps(t), dtmin))

"""
    infinite_or_giant(u)

`true` if any component of `u` is infinite or NaN. This is OrdinaryDiffEq's
default instability check (`ODE_DEFAULT_UNSTABLE_CHECK`), which is
`any(!isfinite, u)`. The loop here gives the same answer but does not stop at
the first hit, so it has no branch and is vectorised; the check runs after
every accepted step, on states that are almost always finite.
"""
function infinite_or_giant(x::Array{<:Union{Float64, ComplexF64}})
    found = false
    @inbounds @simd for i in eachindex(x)
        found |= not_finite(x[i])
    end
    return found
end

"""
    not_finite(x)

`!isfinite(x)`, from the bits: a `Float64` is infinite or NaN exactly if all
eleven bits of its exponent are set. A complex number is tested by its parts.
"""
@inline not_finite(x::Float64) = reinterpret(UInt64, x) & 0x7ff0000000000000 == 0x7ff0000000000000
@inline not_finite(z::ComplexF64) = not_finite(real(z)) | not_finite(imag(z))
