# Copied from OrdinaryDiffEqAdamsBashforthMoulton 1.11.0 (src/algorithms.jl,
# src/alg_utils.jl, src/adams_bashforth_moulton_caches.jl), MIT licence; see
# LICENSE.md. Only the caches for in-place problems are kept.

"""
    VCABM3()

The third-order Adams–Bashforth–Moulton predictor–corrector method with
variable step size, as in OrdinaryDiffEq: two evaluations of `f` per step.
The first two steps are taken with `BS3`.
"""
struct VCABM3 <: LeanAlgorithm end

"""
    VCABM4()

The fourth-order Adams–Bashforth–Moulton predictor–corrector method with
variable step size, as in OrdinaryDiffEq: two evaluations of `f` per step.
The first three steps are taken with `RK4`.
"""
struct VCABM4 <: LeanAlgorithm end

"""
    VCABM5()

The fifth-order Adams–Bashforth–Moulton predictor–corrector method with
variable step size, as in OrdinaryDiffEq: two evaluations of `f` per step.
The first four steps are taken with `RK4`.
"""
struct VCABM5 <: LeanAlgorithm end

"""
    VCABM()

The Adams–Bashforth–Moulton predictor–corrector method with variable step
size and variable order, from 1 to 12, as in OrdinaryDiffEq: two evaluations
of `f` per step. It starts at order 1 and chooses the order by the rules of
Shampine's DDEABM, so it needs no other method for its first steps.
"""
struct VCABM <: LeanAlgorithm end

"""
    alg_order(::VCABM3)

VCABM3 has order 3, VCABM4 order 4 and VCABM5 order 5. For the
variable-order `VCABM` the value 1 is a placeholder, as in OrdinaryDiffEq;
its current order is `get_current_alg_order`.
"""
alg_order(alg::VCABM3) = 3
alg_order(alg::VCABM4) = 4
alg_order(alg::VCABM5) = 5
alg_order(alg::VCABM) = 1  #dummy value

"""
    isstandard(::VCABM)

`VCABM` uses the integral step-size controller (see `controller.jl`).
"""
isstandard(alg::VCABM) = true

"""
    get_current_alg_order(::VCABM, cache)

The order `VCABM` currently runs at.
"""
get_current_alg_order(alg::VCABM, cache) = cache.order

"""
    get_current_adaptive_order(::VCABM, cache)

The order that sets the exponent of `VCABM`'s step-size controller: its
current order.
"""
get_current_adaptive_order(alg::VCABM, cache) = cache.order

"""
    VCABM3Cache

Work arrays of VCABM3. `fsalfirst` and `k4` hold `f(uprev)` and `f(u)`;
`bs3cache` is the cache of the BS3 steps that start the method. `dts` holds
the last three step sizes, newest first, and `step` counts the steps taken
so far, up to the order. `ϕ_n`, `ϕstar_n`, `ϕstar_nm1` and `ϕ_np1` are the
divided differences and `β`, `c`, `g` the coefficients of `adams_utils.jl`.
`utilde` is the error estimate, `atmp` the scaled error and `tmp` scratch
space; `tab` holds BS3's coefficients, and `thread` selects FastBroadcast's
serial kernel.
"""
mutable struct VCABM3Cache{uType, TabType, bs3Type}
    fsalfirst::uType
    bs3cache::bs3Type
    k4::uType
    ϕstar_nm1::Vector{uType}
    dts::Vector{Float64}
    c::Matrix{Float64}
    g::Vector{Float64}
    ϕ_n::Vector{uType}
    ϕ_np1::Vector{uType}
    ϕstar_n::Vector{uType}
    β::Vector{Float64}
    order::Int
    atmp::uType
    tmp::uType
    utilde::uType
    tab::TabType
    step::Int
    thread::FastBroadcast.False
end

"""
    alg_cache(::VCABM3, u, rate_prototype, uprev)

Allocate the work arrays of VCABM3 for states like `u`.
"""
function alg_cache(alg::VCABM3, u, rate_prototype, uprev)
    t = dt = 0.0
    tab = BS3ConstantCache(constvalue(recursive_unitless_bottom_eltype(u)), Float64)
    bs3cache = alg_cache(BS3(), u, rate_prototype, uprev)
    fsalfirst = zero(rate_prototype)
    k4 = zero(rate_prototype)
    dts = fill(zero(dt), 3)
    c = fill(zero(t), 4, 4)
    g = fill(zero(t), 4)
    ϕ_n = Vector{typeof(rate_prototype)}(undef, 3)
    ϕstar_nm1 = Vector{typeof(rate_prototype)}(undef, 3)
    ϕstar_n = Vector{typeof(rate_prototype)}(undef, 3)
    ϕ_np1 = Vector{typeof(rate_prototype)}(undef, 4)
    for i in 1:3
        ϕ_n[i] = zero(rate_prototype)
        ϕstar_nm1[i] = zero(rate_prototype)
        ϕstar_n[i] = zero(rate_prototype)
    end
    for i in 1:4
        ϕ_np1[i] = zero(rate_prototype)
    end
    β = fill(zero(t), 3)
    order = 3
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    tmp = zero(u)
    utilde = zero(u)
    return VCABM3Cache(
        fsalfirst, bs3cache, k4, ϕstar_nm1, dts, c, g, ϕ_n, ϕ_np1,
        ϕstar_n, β, order, atmp, tmp, utilde, tab, 1, FastBroadcast.False()
    )
end

"""
    VCABM4Cache

Work arrays of VCABM4, as `VCABM3Cache`, with four step sizes in `dts` and
`rk4cache`, the cache of the RK4 steps that start the method.
"""
mutable struct VCABM4Cache{uType, rk4cacheType}
    fsalfirst::uType
    rk4cache::rk4cacheType
    k4::uType
    ϕstar_nm1::Vector{uType}
    dts::Vector{Float64}
    c::Matrix{Float64}
    g::Vector{Float64}
    ϕ_n::Vector{uType}
    ϕ_np1::Vector{uType}
    ϕstar_n::Vector{uType}
    β::Vector{Float64}
    order::Int
    atmp::uType
    tmp::uType
    utilde::uType
    step::Int
    thread::FastBroadcast.False
end

"""
    alg_cache(::VCABM4, u, rate_prototype, uprev)

Allocate the work arrays of VCABM4 for states like `u`.
"""
function alg_cache(alg::VCABM4, u, rate_prototype, uprev)
    t = dt = 0.0
    rk4cache = alg_cache(RK4(), u, rate_prototype, uprev)
    fsalfirst = zero(rate_prototype)
    k4 = zero(rate_prototype)
    dts = fill(zero(dt), 4)
    c = fill(zero(t), 5, 5)
    g = fill(zero(t), 5)
    ϕ_n = Vector{typeof(rate_prototype)}(undef, 4)
    ϕstar_nm1 = Vector{typeof(rate_prototype)}(undef, 4)
    ϕstar_n = Vector{typeof(rate_prototype)}(undef, 4)
    ϕ_np1 = Vector{typeof(rate_prototype)}(undef, 5)
    for i in 1:4
        ϕ_n[i] = zero(rate_prototype)
        ϕstar_nm1[i] = zero(rate_prototype)
        ϕstar_n[i] = zero(rate_prototype)
    end
    for i in 1:5
        ϕ_np1[i] = zero(rate_prototype)
    end
    β = fill(zero(t), 4)
    order = 4
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    tmp = zero(u)
    utilde = zero(u)
    return VCABM4Cache(
        fsalfirst, rk4cache, k4, ϕstar_nm1, dts, c, g, ϕ_n, ϕ_np1,
        ϕstar_n, β, order, atmp, tmp, utilde, 1, FastBroadcast.False()
    )
end

"""
    VCABM5Cache

Work arrays of VCABM5, as `VCABM4Cache`, with five step sizes in `dts`.
"""
mutable struct VCABM5Cache{uType, rk4cacheType}
    fsalfirst::uType
    rk4cache::rk4cacheType
    k4::uType
    ϕstar_nm1::Vector{uType}
    dts::Vector{Float64}
    c::Matrix{Float64}
    g::Vector{Float64}
    ϕ_n::Vector{uType}
    ϕ_np1::Vector{uType}
    ϕstar_n::Vector{uType}
    β::Vector{Float64}
    order::Int
    atmp::uType
    tmp::uType
    utilde::uType
    step::Int
    thread::FastBroadcast.False
end

"""
    alg_cache(::VCABM5, u, rate_prototype, uprev)

Allocate the work arrays of VCABM5 for states like `u`.
"""
function alg_cache(alg::VCABM5, u, rate_prototype, uprev)
    t = dt = 0.0
    rk4cache = alg_cache(RK4(), u, rate_prototype, uprev)
    fsalfirst = zero(rate_prototype)
    k4 = zero(rate_prototype)
    dts = fill(zero(dt), 5)
    c = fill(zero(t), 6, 6)
    g = fill(zero(t), 6)
    ϕ_n = Vector{typeof(rate_prototype)}(undef, 5)
    ϕstar_nm1 = Vector{typeof(rate_prototype)}(undef, 5)
    ϕstar_n = Vector{typeof(rate_prototype)}(undef, 5)
    ϕ_np1 = Vector{typeof(rate_prototype)}(undef, 6)
    for i in 1:5
        ϕ_n[i] = zero(rate_prototype)
        ϕstar_nm1[i] = zero(rate_prototype)
        ϕstar_n[i] = zero(rate_prototype)
    end
    for i in 1:6
        ϕ_np1[i] = zero(rate_prototype)
    end
    β = fill(zero(t), 5)
    order = 5
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    tmp = zero(u)
    utilde = zero(u)
    return VCABM5Cache(
        fsalfirst, rk4cache, k4, ϕstar_nm1, dts, c, g, ϕ_n, ϕ_np1,
        ϕstar_n, β, order, atmp, tmp, utilde, 1, FastBroadcast.False()
    )
end

"""
    VCABMCache

Work arrays of the variable-order VCABM, sized for its largest order 12.
Beyond the fields of `VCABM3Cache`: `order` is the current order and
`max_order` its bound; `ξ` and `ξ0` are sums of step sizes kept between
`ϕ_and_ϕstar!` and `expand_ϕ_and_ϕstar!`; `utildem2`, `utildem1` and
`utildep1` are the error estimates of the orders two below, one below and
one above the current one, and `atmpm2`, `atmpm1`, `atmpp1` their scaled
errors.
"""
mutable struct VCABMCache{uType}
    fsalfirst::uType
    k4::uType
    ϕstar_nm1::Vector{uType}
    dts::Vector{Float64}
    c::Matrix{Float64}
    g::Vector{Float64}
    ϕ_n::Vector{uType}
    ϕ_np1::Vector{uType}
    ϕstar_n::Vector{uType}
    β::Vector{Float64}
    order::Int
    max_order::Int
    atmp::uType
    tmp::uType
    ξ::Float64
    ξ0::Float64
    utilde::uType
    utildem1::uType
    utildem2::uType
    utildep1::uType
    atmpm1::uType
    atmpm2::uType
    atmpp1::uType
    step::Int
    thread::FastBroadcast.False
end

"""
    alg_cache(::VCABM, u, rate_prototype, uprev)

Allocate the work arrays of VCABM for states like `u`.
"""
function alg_cache(alg::VCABM, u, rate_prototype, uprev)
    t = dt = 0.0
    fsalfirst = zero(rate_prototype)
    k4 = zero(rate_prototype)
    dts = fill(zero(dt), 13)
    c = fill(zero(t), 13, 13)
    g = fill(zero(t), 13)
    ϕ_n = Vector{typeof(rate_prototype)}(undef, 13)
    ϕstar_nm1 = Vector{typeof(rate_prototype)}(undef, 13)
    ϕstar_n = Vector{typeof(rate_prototype)}(undef, 13)
    ϕ_np1 = Vector{typeof(rate_prototype)}(undef, 14)
    for i in 1:13
        ϕ_n[i] = zero(rate_prototype)
        ϕstar_nm1[i] = zero(rate_prototype)
        ϕstar_n[i] = zero(rate_prototype)
    end
    for i in 1:14
        ϕ_np1[i] = zero(rate_prototype)
    end
    β = fill(zero(t), 13)
    order = 1
    max_order = 12
    tmp = zero(u)
    ξ = zero(dt)
    ξ0 = zero(dt)
    utilde = zero(u)
    utildem2 = zero(u)
    utildem1 = zero(u)
    utildep1 = zero(u)
    atmp = similar(u, eltype(u))
    fill!(atmp, false)
    atmpm1 = similar(u, eltype(u))
    fill!(atmpm1, false)
    atmpm2 = similar(u, eltype(u))
    fill!(atmpm2, false)
    atmpp1 = similar(u, eltype(u))
    fill!(atmpp1, false)
    return VCABMCache(
        fsalfirst, k4, ϕstar_nm1, dts, c, g, ϕ_n, ϕ_np1, ϕstar_n, β, order,
        max_order, atmp, tmp, ξ, ξ0, utilde, utildem1, utildem2, utildep1, atmpm1,
        atmpm2, atmpp1, 1, FastBroadcast.False()
    )
end

"""
    VariableCoefficientCache

The caches of the Adams methods with variable step size. All keep
`f(uprev)` in `fsalfirst` and `f(u)` in `k4`.
"""
const VariableCoefficientCache = Union{VCABM3Cache, VCABM4Cache, VCABM5Cache, VCABMCache}

"""
    get_fsalfirstlast(cache::VariableCoefficientCache)

The arrays that hold `f(uprev)` and `f(u)`: `fsalfirst` and `k4`.
"""
get_fsalfirstlast(cache::VariableCoefficientCache) = (cache.fsalfirst, cache.k4)

"""
    exchange_fsal!(integrator, cache::VariableCoefficientCache)

Exchange `fsalfirst` and `k4`, the arrays that hold `f(uprev)` and `f(u)`.
"""
function exchange_fsal!(integrator, cache::VariableCoefficientCache)
    cache.fsalfirst, cache.k4 = cache.k4, cache.fsalfirst
    integrator.fsalfirst = integrator.k[1] = cache.fsalfirst
    integrator.fsallast = integrator.k[2] = cache.k4
    return nothing
end
