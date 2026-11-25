# Through the SciMLBase extension, LeanODE.solve runs OrdinaryDiffEq's methods.

@testset "OrdinaryDiffEq methods through the extension" begin
    (; f!, u0, tspan, p) = PROBLEMS.lorenz
    for alg in (OrdinaryDiffEqAdamsBashforthMoulton.VCABM3(), OrdinaryDiffEqAdamsBashforthMoulton.VCABM5())
        via_leanode = LeanODE.solve(LeanODE.ODEProblem(f!, u0, tspan, p), alg; reltol = 1e-8, abstol = 1e-8)
        direct = SciMLBase.solve(SciMLBase.ODEProblem(f!, u0, tspan, p), alg; reltol = 1e-8, abstol = 1e-8)
        @test via_leanode isa LeanODE.ODESolution
        @test identical(via_leanode, direct)
    end
    early = LeanODE.solve(LeanODE.ODEProblem(f!, u0, tspan, p), OrdinaryDiffEqTsit5.Tsit5(); maxiters = 10, verbose = false)
    @test early.retcode == ReturnCode.MaxIters
end
