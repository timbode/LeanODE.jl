module LeanODE

include("vendor/FastPower.jl")
using .FastPower: fastpower

export ODEProblem, ODESolution, ReturnCode, solve
export Tsit5, RK4, BS3, VCABM, VCABM3, VCABM4, VCABM5

include("types.jl")
include("misc_utils.jl")
include("defaults.jl")
include("kernels.jl")
include("integrator.jl")
include("initdt.jl")
include("controller.jl")
include("saving.jl")
include("solve.jl")
include("tsit5_tableaus.jl")
include("tsit5.jl")
include("rk4.jl")
include("bs3.jl")
include("adams_caches.jl")
include("adams_utils.jl")
include("adams_perform_step.jl")
include("hermite.jl")
include("precompile.jl")

end # module LeanODE
