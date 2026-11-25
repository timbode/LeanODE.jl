# Copied from OrdinaryDiffEqAdamsBashforthMoulton 1.11.0 (src/adams_utils.jl),
# MIT licence; see LICENSE.md. Only the branches for in-place caches are kept.
#
# The variable-step Adams methods in the form of Hairer, Nørsett and Wanner,
# Solving ODEs I, Sec. III.5. With the last step sizes dts[1] = tₙ₊₁ - tₙ,
# dts[2] = tₙ - tₙ₋₁, …, the methods work with the modified divided
# differences ϕᵢ(n) of the last derivatives fₙ, fₙ₋₁, … (formulas 5.9):
#
#   ϕ₁(n) = fₙ,   ϕᵢ(n) = ϕᵢ₋₁(n) - ϕ*ᵢ₋₁(n - 1),   ϕ*ᵢ(n) = βᵢ(n) ϕᵢ(n),
#
# and the step from tₙ to tₙ₊₁ is (formulas 5.7 and 5.10)
#
#   predictor   pₙ₊₁ = uₙ + Σᵢ gᵢ ϕ*ᵢ(n)        (the gᵢ include the step size)
#   corrector   uₙ₊₁ = pₙ₊₁ + gₖ ϕₖ₊₁(n + 1)    with ϕ(n + 1) from f(pₙ₊₁).
#
# In the code, `ϕ_n[i]` is ϕᵢ(n), `ϕstar_n[i]` is ϕ*ᵢ(n), `ϕstar_nm1[i]` is
# ϕ*ᵢ(n - 1) and `ϕ_np1[i]` is ϕᵢ(n + 1).

"""
    ϕ_and_ϕstar!(cache, du, k)

Compute the divided differences `ϕ_n[1:k]` and `ϕstar_n[1:k]` of order up to
`k` from the newest derivative `du = fₙ`, the previous `ϕstar_nm1` and the
step sizes `cache.dts`, and the coefficients `β[1:k]`.
"""
function ϕ_and_ϕstar!(cache, du, k)
    return @inbounds begin
        (; dts, ϕstar_nm1, ϕ_n, ϕstar_n, β) = cache
        ξ = dt = dts[1]
        ξ0 = zero(dt)
        β[1] = one(dt)
        ϕ_n[1] .= du
        ϕstar_n[1] .= du
        for i in 2:k
            ξ0 += dts[i]
            β[i] = β[i - 1] * ξ / ξ0
            ξ += dts[i]
            @.. broadcast = false ϕ_n[i] = ϕ_n[i - 1] - ϕstar_nm1[i - 1]
            @.. broadcast = false ϕstar_n[i] = β[i] * ϕ_n[i]
        end
    end # inbounds
end

"""
    ϕ_and_ϕstar!(cache::VCABMCache, du, k)

As the general method; it also stores the sums of step sizes `ξ` and `ξ0`,
from which `expand_ϕ_and_ϕstar!` continues to the next order.
"""
function ϕ_and_ϕstar!(cache::VCABMCache, du, k)
    return @inbounds begin
        (; dts, ϕstar_nm1, ϕ_n, ϕstar_n, β) = cache
        ξ = dt = dts[1]
        ξ0 = zero(dt)
        β[1] = one(dt)
        ϕ_n[1] .= du
        ϕstar_n[1] .= du
        for i in 2:k
            ξ0 += dts[i]
            β[i] = β[i - 1] * ξ / ξ0
            ξ += dts[i]
            @.. broadcast = false ϕ_n[i] = ϕ_n[i - 1] - ϕstar_nm1[i - 1]
            @.. broadcast = false ϕstar_n[i] = β[i] * ϕ_n[i]
        end
        cache.ξ = ξ
        cache.ξ0 = ξ0
    end # inbounds
end

"""
    expand_ϕ_and_ϕstar!(cache, i)

Extend `ϕ_n`, `ϕstar_n` and `β` by one order, to `i`, continuing from the
sums `cache.ξ` and `cache.ξ0` that `ϕ_and_ϕstar!` left.
"""
function expand_ϕ_and_ϕstar!(cache, i)
    (; ξ, ξ0, β, dts, ϕstar_nm1, ϕ_n, ϕstar_n) = cache
    ξ0 += dts[i]
    β[i] = β[i - 1] * ξ / ξ0
    @.. broadcast = false ϕ_n[i] = ϕ_n[i - 1] - ϕstar_nm1[i - 1]
    @.. broadcast = false ϕstar_n[i] = β[i] * ϕ_n[i]
    return nothing
end

"""
    ϕ_np1!(cache, du_np1, k)

Compute the divided differences `ϕ_np1[1:k]` at the new point from the
derivative `du_np1` there and `ϕstar_n`.
"""
function ϕ_np1!(cache, du_np1, k)
    return @inbounds begin
        (; ϕ_np1, ϕstar_n) = cache
        for i in 1:k
            if i != 1
                @.. broadcast = false ϕ_np1[i] = ϕ_np1[i - 1] - ϕstar_n[i - 1]
            else
                ϕ_np1[i] .= du_np1
            end
        end
    end #inbounds
end

"""
    g_coefs!(cache, k)

Compute the coefficients `g[1:k]` of the predictor and corrector from the
step sizes `cache.dts`, by the recurrence of formulas 5.9 and 5.10 (`c` is its
work matrix). They include the factor `dt`.
"""
function g_coefs!(cache, k)
    return @inbounds begin
        (; dts, c, g) = cache
        ξ = dt = dts[1]
        for i in 1:k
            if i > 2
                ξ += dts[i - 1]
            end
            for q in 1:(k - (i - 1))
                if i > 2
                    c[i, q] = muladd(-dt / ξ, c[i - 1, q + 1], c[i - 1, q])
                elseif i == 1
                    c[i, q] = inv(q)
                elseif i == 2
                    c[i, q] = inv(q * (q + 1))
                end
            end # q
            g[i] = c[i, 1] * dt
        end # i
    end # inbounds
end

"""
    γstar

The coefficients γ*ⱼ of the implicit Adams methods for constant step size
(index `j + 1`). The variable-order method uses them to estimate the error
that one order higher would have made.
"""
const γstar = [
    1,
    -1 / 2,
    -1 / 12,
    -1 / 24,
    -19 / 720,
    -3 / 160,
    -863 / 60480,
    -275 / 24192,
    -33953 / 3628800,
    -0.00789255,
    -0.00678585,
    -0.00592406,
    -0.00523669,
    -0.0046775,
    -0.00421495,
    -0.0038269,
]
