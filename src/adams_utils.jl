# After OrdinaryDiffEqAdamsBashforthMoulton 1.11.0 (src/adams_utils.jl), MIT
# licence; see LICENSE.md. The coefficients are computed as upstream; the
# loops over the state are LeanODE's own.
#
# The variable-step Adams methods in the form of Hairer, Nørsett and Wanner,
# Solving ODEs I, Sec. III.5. With the last step sizes dts[1] = tₙ₊₁ - tₙ,
# dts[2] = tₙ - tₙ₋₁, …, the methods work with the modified divided
# differences ϕᵢ(n) of the last derivatives fₙ, fₙ₋₁, … (formulas 5.9):
#
#   ϕ₁(n) = fₙ,   ϕᵢ(n) = ϕᵢ₋₁(n) - ϕ*ᵢ₋₁(n - 1),   ϕ*ᵢ(n) = βᵢ(n) ϕᵢ(n),
#
# and the step from tₙ to tₙ₊₁ at order k is (formulas 5.7 and 5.10)
#
#   predictor   pₙ₊₁ = uₙ + Σᵢ₌₁ᵏ⁻¹ gᵢ ϕ*ᵢ(n)    (the gᵢ include the step size)
#   corrector   uₙ₊₁ = pₙ₊₁ + gₖ ϕₖ(n + 1)       with ϕ(n + 1) from f(pₙ₊₁)
#   error       (gₖ₊₁ - gₖ) ϕₖ₊₁(n + 1).
#
# In the code, `ϕstar_n[i]` is ϕ*ᵢ(n) and `ϕstar_nm1[i]` is ϕ*ᵢ(n - 1). The
# differences ϕᵢ(n) and ϕᵢ(n + 1) are needed only while the next one is
# formed, so they are not stored, except ϕₖ(n) in `ϕ_k`.
#
# OrdinaryDiffEq forms every difference in a pass of its own over the whole
# state, about 110 passes per step at order 5. The functions here form all
# differences of a step in three sweeps. A sweep takes the state in blocks of
# `BLOCK` components, so that its passes run on data in the processor's
# first-level cache, and each pass advances up to four orders, so that the
# running values stay in registers. Per component the arithmetic of the state
# is upstream's, operation for operation. The error norms differ in the last
# bits: they are summed in a different order and divide once per component.

"""
    BLOCK

The number of components the sweeps process at a time. A pass reads and
writes about ten arrays, 4 kB of each per block, which fits into a
first-level cache of 48 kB. On an Apple M4, 512 and 1024 were about equally
fast and 256 and 4096 slower.
"""
const BLOCK = 512

"""
    blocks(u)

The index ranges `1:BLOCK`, `BLOCK+1:2BLOCK`, … that cover the array `u`.
"""
blocks(u) = (first:min(first + BLOCK - 1, length(u)) for first in 1:BLOCK:length(u))

"""
    β_coefs!(cache, k)

Compute the coefficients `β[1:k]` of formulas 5.9 from the step sizes
`cache.dts`. Returns `(ξ, ξ0)`, the sums of step sizes the recurrence ends
with; `β[k + 1] = β[k] ξ / (ξ0 + dts[k + 1])` continues it by one order.
"""
function β_coefs!(cache, k)
    (; dts, β) = cache
    ξ = dt = dts[1]
    ξ0 = zero(dt)
    β[1] = one(dt)
    @inbounds for i in 2:k
        ξ0 += dts[i]
        β[i] = β[i - 1] * ξ / ξ0
        ξ += dts[i]
    end
    return ξ, ξ0
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

"""
    Shifted(data, offset)

A block vector addressed by the indices of the block it stands for:
`shifted[i]` is `data[i - offset]`.
"""
struct Shifted{V}
    data::V
    offset::Int
end

Base.@propagate_inbounds Base.getindex(shifted::Shifted, i) = shifted.data[i - shifted.offset]
Base.@propagate_inbounds function Base.setindex!(shifted::Shifted, value, i)
    return shifted.data[i - shifted.offset] = value
end

"""
    with_count(f, count, args...)

Call `f(Val(count), args...)` for a `count` from 0 to 4, in a way that lets
the compiler know the count in each case. The sweeps use it to advance
`count` orders in one pass.
"""
@inline function with_count(f::F, count, args...) where {F}
    return count == 4 ? f(Val(4), args...) : count == 3 ? f(Val(3), args...) :
        count == 2 ? f(Val(2), args...) : count == 1 ? f(Val(1), args...) : f(Val(0), args...)
end

# ------------------------------------------------ first sweep: predictor
#
# The predictor uses the differences of the orders 1 to `terms`; the orders
# after that, up to k, are formed without it. The two kinds run in passes of
# their own, so that each loop contains the predictor's `muladd` either for
# every order or for none: inside a conditional the compiler does not always
# fuse it, and the states would then differ from upstream's in the last bit.

"""
    advance(ϕ, u, i, (ϕstar, ϕstar_previous, β, g), more...)

For component `i`, advance the divided difference `ϕ` by one order per term:
`ϕ - ϕstar_previous[i]`, stored times `β` in `ϕstar[i]` and added times `g`
to the predictor `u`. Returns the last `ϕ` and `u`.
"""
@inline advance(ϕ, u, i) = (ϕ, u)
Base.@propagate_inbounds function advance(ϕ, u, i, (ϕstar, ϕstar_previous, β, g), more...)
    ϕ -= ϕstar_previous[i]
    scaled = β * ϕ
    ϕstar[i] = scaled
    return advance(ϕ, muladd(g, scaled, u), i, more...)
end

"""
    advance_plain(ϕ, i, (ϕstar, ϕstar_previous, β), more...)

As `advance`, without a predictor. Returns the last `ϕ`.
"""
@inline advance_plain(ϕ, i) = ϕ
Base.@propagate_inbounds function advance_plain(ϕ, i, (ϕstar, ϕstar_previous, β), more...)
    ϕ -= ϕstar_previous[i]
    ϕstar[i] = β * ϕ
    return advance_plain(ϕ, i, more...)
end

"""
    predictor_terms(cache, j, Val(N))

The arguments of `advance` for the `N` orders from `j` on.
"""
@inline function predictor_terms(cache, j, ::Val{N}) where {N}
    (; ϕstar_n, ϕstar_nm1, β, g) = cache
    return ntuple(Val(N)) do m
        order = j + m - 1
        @inbounds (ϕstar_n[order], ϕstar_nm1[order - 1], β[order], g[order])
    end
end

"""
    difference_terms(cache, j, Val(N))

The arguments of `advance_plain` for the `N` orders from `j` on.
"""
@inline function difference_terms(cache, j, ::Val{N}) where {N}
    (; ϕstar_n, ϕstar_nm1, β) = cache
    return ntuple(Val(N)) do m
        order = j + m - 1
        @inbounds (ϕstar_n[order], ϕstar_nm1[order - 1], β[order])
    end
end

"""
    start_predictor!(Val(N), block, u, uprev, du, cache)

On the components `block`: start the divided differences with the first,
`du`, and the predictor with `u = uprev + g[1] du`, then advance both through
the orders 2 to `N + 1`. Leaves the last difference in `cache.ϕ_k`.
"""
@inline function start_predictor!(count::Val, block, u, uprev, du, cache)
    return start_predictor!(block, u, uprev, du, cache, predictor_terms(cache, 2, count)...)
end

@inline function start_predictor!(block::AbstractUnitRange, u, uprev, du, cache, terms::Vararg{Any, N}) where {N}
    (; ϕ_k) = cache
    ϕstar₁ = cache.ϕstar_n[1]
    g₁ = cache.g[1]
    @inbounds @simd ivdep for i in block
        ϕ = du[i]
        ϕstar₁[i] = ϕ
        ϕ, predictor = advance(ϕ, muladd(g₁, ϕ, uprev[i]), i, terms...)
        ϕ_k[i] = ϕ
        u[i] = predictor
    end
    return nothing
end

"""
    continue_predictor!(Val(N), block, u, cache, j)

On the components `block`: advance the difference in `cache.ϕ_k` and the
predictor `u` through the `N` orders from `j` on.
"""
@inline function continue_predictor!(count::Val, block, u, cache, j)
    return continue_predictor!(block, u, cache, predictor_terms(cache, j, count)...)
end

@inline function continue_predictor!(block::AbstractUnitRange, u, cache, terms::Vararg{Any, N}) where {N}
    (; ϕ_k) = cache
    @inbounds @simd ivdep for i in block
        ϕ, predictor = advance(ϕ_k[i], u[i], i, terms...)
        ϕ_k[i] = ϕ
        u[i] = predictor
    end
    return nothing
end

"""
    start_differences!(Val(N), block, du, cache)

On the components `block`: start the divided differences with the first,
`du`, and advance them through the orders 2 to `N + 1`, without a predictor.
Leaves the last difference in `cache.ϕ_k`.
"""
@inline function start_differences!(count::Val, block, du, cache)
    return start_differences!(block, du, cache, difference_terms(cache, 2, count)...)
end

@inline function start_differences!(block::AbstractUnitRange, du, cache, terms::Vararg{Any, N}) where {N}
    (; ϕ_k) = cache
    ϕstar₁ = cache.ϕstar_n[1]
    @inbounds @simd ivdep for i in block
        ϕ = du[i]
        ϕstar₁[i] = ϕ
        ϕ_k[i] = advance_plain(ϕ, i, terms...)
    end
    return nothing
end

"""
    continue_differences!(Val(N), block, cache, j)

On the components `block`: advance the difference in `cache.ϕ_k` through the
`N` orders from `j` on, without a predictor.
"""
@inline function continue_differences!(count::Val, block, cache, j)
    return continue_differences!(block, cache, difference_terms(cache, j, count)...)
end

@inline function continue_differences!(block::AbstractUnitRange, cache, terms::Vararg{Any, N}) where {N}
    (; ϕ_k) = cache
    @inbounds @simd ivdep for i in block
        ϕ_k[i] = advance_plain(ϕ_k[i], i, terms...)
    end
    return nothing
end

"""
    predict!(u, uprev, du, cache, k, terms)

The first sweep of a step: form the divided differences `ϕstar_n[1:k]` of
`du = f(uprev)` and the predictor `u = uprev + Σᵢ g[i] ϕstar_n[i]` over the
first `terms` of them. With `terms = 0`, `u` is not touched. Leaves `ϕₖ(n)`
in `cache.ϕ_k`.
"""
function predict!(u, uprev, du, cache, k, terms)
    for block in blocks(u)
        if terms ≥ 1
            done = min(terms, 4)
            with_count(start_predictor!, done - 1, block, u, uprev, du, cache)
            while done < terms
                count = min(terms - done, 4)
                with_count(continue_predictor!, count, block, u, cache, done + 1)
                done += count
            end
        else
            done = min(k, 4)
            with_count(start_differences!, done - 1, block, du, cache)
        end
        while done < k
            count = min(k - done, 4)
            with_count(continue_differences!, count, block, cache, done + 1)
            done += count
        end
    end
    return nothing
end

# ---------------------------------------- second sweep: corrector, errors

"""
    minus(ψ, i, ϕstar₁, ϕstar₂, …)

`ψ - ϕstar₁[i] - ϕstar₂[i] - …`, subtracted in this order.
"""
@inline minus(ψ, i) = ψ
Base.@propagate_inbounds minus(ψ, i, ϕstar, more...) = minus(ψ - ϕstar[i], i, more...)

"""
    subtract!(target, source, block, ϕstar₁, ϕstar₂, …)

On the components `block`: `target = source - ϕstar₁ - ϕstar₂ - …`. With
`source = ϕⱼ(n + 1)` and the divided differences `ϕ*ⱼ(n), ϕ*ⱼ₊₁(n), …`, the
result is a difference of higher order at the new point. `target` may be
`source`.
"""
@inline function subtract!(target, source, block, ϕstars::Vararg{Any, N}) where {N}
    @inbounds @simd ivdep for i in block
        target[i] = minus(source[i], i, ϕstars...)
    end
    return nothing
end

"""
    subtract!(Val(N), target, source, block, ϕstars, j)

`subtract!` with the `N` arrays `ϕstars[j]`, `ϕstars[j + 1]`, ….
"""
@inline function subtract!(::Val{N}, target, source, block, ϕstars, j) where {N}
    return subtract!(target, source, block, ntuple(m -> @inbounds(ϕstars[j + m - 1]), Val(N))...)
end

"""
    subtract_range!(ψ, du, block, ϕstars, last)

On the components `block`: `ψ = du - ϕstars[1] - … - ϕstars[last]`, in
passes of up to four subtractions. With `du = f` at the new point this is
`ϕⱼ(n + 1)` for `j = last + 1`.
"""
@inline function subtract_range!(ψ, du, block, ϕstars, last)
    done = min(last, 4)
    with_count(subtract!, done, ψ, du, block, ϕstars, 1)
    while done < last
        count = min(last - done, 4)
        with_count(subtract!, count, ψ, ψ, block, ϕstars, done + 1)
        done += count
    end
    return nothing
end

"""
    correct_block!(block, u, uprev, ψ, cache, k, abstol, reltol, lower_orders)

On the components `block`, with `ψ = ϕₖ₋₁(n + 1)`: form `ϕₖ(n + 1)` and
`ϕₖ₊₁(n + 1)`, add the corrector's term to `u`, and return the sums of the
squared scaled error estimates of the orders `k`, `k - 1` and `k - 2` (the
last two are zero without `lower_orders`).
"""
@inline function correct_block!(block, u, uprev, ψ, cache, k, abstol, reltol, lower_orders)
    (; ϕstar_n, g) = cache
    ϕstar_km1 = ϕstar_n[k - 1]
    ϕstar_k = ϕstar_n[k]
    g_k = g[k]
    weight = g[k + 1] - g[k]
    total = total_one_lower = total_two_lower = 0.0
    if lower_orders
        weight_one_lower = g[k] - g[k - 1]
        weight_two_lower = g[k - 1] - g[k - 2]
        @inbounds @simd ivdep for i in block
            ψ_km1 = ψ[i]
            ψ_k = ψ_km1 - ϕstar_km1[i]
            corrected = muladd(g_k, ψ_k, u[i])
            u[i] = corrected
            ψ_kp1 = ψ_k - ϕstar_k[i]
            inverse = inverse_scale(uprev[i], corrected, abstol, reltol)
            total += abs2(weight * ψ_kp1 * inverse)
            total_one_lower += abs2(weight_one_lower * ψ_k * inverse)
            total_two_lower += abs2(weight_two_lower * ψ_km1 * inverse)
        end
    else
        @inbounds @simd ivdep for i in block
            ψ_k = ψ[i] - ϕstar_km1[i]
            corrected = muladd(g_k, ψ_k, u[i])
            u[i] = corrected
            ψ_kp1 = ψ_k - ϕstar_k[i]
            total += abs2(weight * ψ_kp1 * inverse_scale(uprev[i], corrected, abstol, reltol))
        end
    end
    return total, total_one_lower, total_two_lower
end

"""
    correct_first_order!(block, u, uprev, du, cache, abstol, reltol)

`correct_block!` for order 1, where the corrector's term is `g[1] du`
itself. Returns the sum of the squared scaled error estimates.
"""
@inline function correct_first_order!(block, u, uprev, du, cache, abstol, reltol)
    (; g) = cache
    ϕstar₁ = cache.ϕstar_n[1]
    g₁ = g[1]
    weight = g[2] - g[1]
    total = 0.0
    @inbounds @simd ivdep for i in block
        corrected = muladd(g₁, du[i], u[i])
        u[i] = corrected
        ψ₂ = du[i] - ϕstar₁[i]
        total += abs2(weight * ψ₂ * inverse_scale(uprev[i], corrected, abstol, reltol))
    end
    return total
end

"""
    correct!(u, uprev, du, cache, k, abstol, reltol, lower_orders)

The second sweep of a step at order `k`: from `du`, `f` at the predicted
state `u`, form the differences `ϕⱼ(n + 1)`, add the corrector's term
`g[k] ϕₖ(n + 1)` to `u`, and return the scaled norm of the error estimate
`(g[k + 1] - g[k]) ϕₖ₊₁(n + 1)` (see `error_norm`).

With `lower_orders`, which needs `k ≥ 3`, also return the norms of the
estimates for the orders `k - 1` and `k - 2`, `(g[k] - g[k - 1]) ϕₖ(n + 1)`
and `(g[k - 1] - g[k - 2]) ϕₖ₋₁(n + 1)`; otherwise these are zero. The
return value is `(error, error_one_lower, error_two_lower)`.
"""
function correct!(u, uprev, du, cache, k, abstol, reltol, lower_orders)
    total = total_one_lower = total_two_lower = 0.0
    for block in blocks(u)
        if k == 1
            total += correct_first_order!(block, u, uprev, du, cache, abstol, reltol)
        else
            ψ = Shifted(cache.ψ, first(block) - 1)
            subtract_range!(ψ, du, block, cache.ϕstar_n, k - 2)      # ϕₖ₋₁(n + 1)
            sums = correct_block!(block, u, uprev, ψ, cache, k, abstol, reltol, lower_orders)
            total += sums[1]
            total_one_lower += sums[2]
            total_two_lower += sums[3]
        end
    end
    return root_mean(total, u), root_mean(total_one_lower, u), root_mean(total_two_lower, u)
end

# ------------------------------- third sweep: the error of the next order

"""
    expand!(ϕstar, ϕ, ϕstar_previous, β, block)

On the components `block`: `ϕstar = β (ϕ - ϕstar_previous)`, the divided
difference one order above the last one formed.
"""
@inline function expand!(ϕstar, ϕ, ϕstar_previous, β, block)
    @inbounds @simd ivdep for i in block
        ϕstar[i] = β * (ϕ[i] - ϕstar_previous[i])
    end
    return nothing
end

"""
    scaled_squares(weight, ψ, uprev, u, abstol, reltol, block)

The sum over the components `block` of the squared scaled error
`|weight ψ / (abstol + max(|uprev|, |u|) reltol)|²`.
"""
@inline function scaled_squares(weight, ψ, uprev, u, abstol, reltol, block)
    total = 0.0
    @inbounds @simd ivdep for i in block
        total += abs2(weight * ψ[i] * inverse_scale(uprev[i], u[i], abstol, reltol))
    end
    return total
end

"""
    higher_order_error!(cache, du, uprev, u, k, dt, abstol, reltol)

The third sweep, for the variable-order method after an accepted step at
order `k`: extend the divided differences by `ϕstar_n[k + 1]`, form
`ϕₖ₊₂(n + 1)` from `du = f(u)`, and return the scaled norm of
`dt γ*ₖ₊₁ ϕₖ₊₂(n + 1)`, the error estimate of order `k + 1`.
"""
function higher_order_error!(cache, du, uprev, u, k, dt, abstol, reltol)
    (; ϕstar_nm1, ϕstar_n, ϕ_k, β, dts) = cache
    β[k + 1] = β[k] * cache.ξ / (cache.ξ0 + dts[k + 1])
    weight = dt * γstar[(k + 1) + 1]
    total = 0.0
    @inbounds for block in blocks(u)
        ψ = Shifted(cache.ψ, first(block) - 1)
        expand!(ϕstar_n[k + 1], ϕ_k, ϕstar_nm1[k], β[k + 1], block)
        subtract_range!(ψ, du, block, ϕstar_n, k + 1)
        total += scaled_squares(weight, ψ, uprev, u, abstol, reltol, block)
    end
    return root_mean(total, u)
end
