# Copied from OrdinaryDiffEqCore 3.33.1 (src/misc_utils.jl, src/OrdinaryDiffEqCore.jl)
# and RecursiveArrayTools 3.54.0 (src/utils.jl), MIT licence; see LICENSE.md.

"""
    CompiledFloats

The float types for which the coefficient tables are written as `Float64`
literals; other types convert from `BigFloat` strings.
"""
const CompiledFloats = Union{Float32, Float64}

"""
    @fold function ... end

Mark a function as foldable (`Base.@assume_effects :foldable`), so that a
coefficient table is computed at compile time.
"""
macro fold(arg)
    return esc(:(Base.@assume_effects :foldable $arg))
end

"""
    @OnDemandTableauExtract S T T2

Construct the coefficient table `S(T, T2)` and bind each of its fields to a
local variable of the same name.
"""
macro OnDemandTableauExtract(S_T, T, T2)
    S = getproperty(__module__, S_T)
    s = gensym(:s)
    q = quote
        $s = $S($T, $T2)
    end
    fn = fieldnames(S)
    for n in fn
        push!(q.args, Expr(:(=), n, Expr(:call, :getfield, s, QuoteNode(n))))
    end
    return esc(q)
end

"""
    constvalue(T)

The real type of the coefficients for values of type `T`: `T` itself, or its
real part type if `T` is complex.
"""
function constvalue(::Type{T}) where {T}
    return T <: Complex ? real(T) : T
end

"""
    recursive_unitless_bottom_eltype(u)

The element type of the array `u`, as `typeof(one(eltype(u)))`.
"""
recursive_unitless_bottom_eltype(u::AbstractArray{T}) where {T <: Number} = typeof(one(T))
