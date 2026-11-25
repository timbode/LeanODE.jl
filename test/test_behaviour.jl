# LeanODE's own interface, independent of the comparison.

@testset "behaviour" begin
    @testset "accuracy on exponential decay" begin
        problem = LeanODE.ODEProblem(decay!, [1.0, 2.0], (0.0, 1.0), 1.0)
        for alg in ALGORITHMS
            sol = LeanODE.solve(problem, alg; reltol = 1e-10, abstol = 1e-12)
            @test sol.retcode == ReturnCode.Success
            @test sol.t[1] == 0.0 && sol.t[end] == 1.0
            @test sol.u[end] ≈ exp(-1) .* [1.0, 2.0] rtol = 1e-8
        end
    end

    @testset "the initial state is not changed" begin
        u0 = [1.0, 2.0]
        LeanODE.solve(LeanODE.ODEProblem(decay!, u0, (0.0, 1.0), 1.0), LeanODE.Tsit5())
        @test u0 == [1.0, 2.0]
    end

    @testset "saved states are copies" begin
        sol = LeanODE.solve(LeanODE.ODEProblem(decay!, [1.0, 2.0], (0.0, 1.0), 1.0), LeanODE.Tsit5())
        @test all(sol.u[i] !== sol.u[j] for i in eachindex(sol.u), j in eachindex(sol.u) if i < j)
    end

    @testset "return code names" begin
        @test Symbol(ReturnCode.Success) == :Success
        @test Symbol(ReturnCode.MaxIters) == :MaxIters
    end

    @testset "unsupported states are refused" begin
        @test_throws ArgumentError LeanODE.ODEProblem(decay!, [1, 2], (0.0, 1.0), 1.0)
        @test_throws ArgumentError LeanODE.ODEProblem(decay!, Float32[1, 2], (0.0, 1.0), 1.0)
    end

    @testset "fixed steps need dt" begin
        problem = LeanODE.ODEProblem(decay!, [1.0], (0.0, 1.0), 1.0)
        @test_throws ArgumentError LeanODE.solve(problem, LeanODE.RK4(); adaptive = false)
    end
end
