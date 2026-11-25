"""
    LeanODESciMLBaseExt

Lets `LeanODE.solve` take any OrdinaryDiffEq method. Code written against
LeanODE can then still use methods that LeanODE lacks, wherever OrdinaryDiffEq
is loaded, for example in tests and notebooks.
"""
module LeanODESciMLBaseExt

using LeanODE: LeanODE, ODEProblem, ODESolution, ReturnCode, Stats
import SciMLBase

"""
    leanode_retcode(retcode)

LeanODE's return code with the same name as SciMLBase's `retcode`, or
`ReturnCode.Failure` if LeanODE has none of that name.
"""
function leanode_retcode(retcode)
    name = Symbol(retcode)
    for code in instances(ReturnCode.T)
        Symbol(code) == name && return code
    end
    return ReturnCode.Failure
end

"""
    LeanODE.solve(prob::LeanODE.ODEProblem, alg::SciMLBase.AbstractODEAlgorithm; kwargs...)

Solve `prob` with the OrdinaryDiffEq method `alg`. The keywords are passed to
`SciMLBase.solve` unchanged. The result is a LeanODE `ODESolution` holding
OrdinaryDiffEq's saved times and states, work counters and return code.
"""
function LeanODE.solve(prob::ODEProblem, alg::SciMLBase.AbstractODEAlgorithm; kwargs...)
    sciml_problem = SciMLBase.ODEProblem(prob.f, prob.u0, prob.tspan, prob.p)
    sol = SciMLBase.solve(sciml_problem, alg; kwargs...)
    stats = Stats(sol.stats.nf, sol.stats.naccept, sol.stats.nreject)
    return ODESolution(sol.t, sol.u, leanode_retcode(sol.retcode), stats, prob, alg)
end

end # module
