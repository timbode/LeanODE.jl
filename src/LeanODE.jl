module LeanODE

include("vendor/MuladdMacro.jl")
include("vendor/FastPower.jl")
include("vendor/FastBroadcast.jl")
using .MuladdMacro: @muladd
using .FastPower: fastpower
using .FastBroadcast: @..

export ODEProblem, ODESolution, ReturnCode, solve
export Tsit5, RK4

include("types.jl")
include("misc_utils.jl")
include("defaults.jl")
include("integrator.jl")
include("initdt.jl")
include("controller.jl")
include("saving.jl")
include("solve.jl")
include("tsit5_tableaus.jl")
include("tsit5.jl")
include("rk4.jl")

end # module LeanODE
