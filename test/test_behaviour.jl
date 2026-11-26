# LeanODE's own interface, independent of the comparison.

"""
    constant_slope!(du, u, p, t)

Right-hand side of `u' = 1`, whose solution is `u(t) = u(0) + t`.
"""
function constant_slope!(du, u, p, t)
    du .= 1.0
    return nothing
end

"""
    first_step_error(alg, step_size)

The error of one step of size `step_size` on `u' = -u` from `u(0) = 1`. The
tolerances are loose, so that the step is accepted as it is.
"""
function first_step_error(alg, step_size)
    problem = LeanODE.ODEProblem(decay!, [1.0], (0.0, 1.0), 1.0)
    integrator = LeanODE.init(problem, alg; dt = step_size, reltol = 1.0e-2, abstol = 1.0e-2)
    lean_step!(integrator)
    @test integrator.t == step_size
    return abs(integrator.u[1] - exp(-integrator.t))
end

"""
    fixed_step_error(alg, dt)

The error at `t = 1` of `u' = -u` from `u(0) = 1`, solved with steps of size
`dt` and without step-size control.
"""
function fixed_step_error(alg, dt)
    problem = LeanODE.ODEProblem(decay!, [1.0], (0.0, 1.0), 1.0)
    return abs(LeanODE.solve(problem, alg; adaptive = false, dt).u[end][1] - exp(-1))
end

@testset "behaviour" begin
    @testset "a constant slope is followed exactly" begin
        problem = LeanODE.ODEProblem(constant_slope!, [5.0], (0.0, 1.0))
        for alg in ALGORITHMS
            @test LeanODE.solve(problem, alg).u[end][1] ≈ 6.0 atol = 1.0e-12
            @test LeanODE.solve(problem, alg; adaptive = false, dt = 0.1).u[end][1] ≈ 6.0 atol = 1.0e-12
        end
    end

    # VCABM starts every solve at order 1, and without step-size control it
    # stays there. OrdinaryDiffEq's order-1 step advances by twice the slope
    # (see vcabm_reference.jl); these checks need no reference to catch that.
    @testset "VCABM at order 1" begin
        @testset "one step has an error of order h²" begin
            for step_size in (1.0e-2, 1.0e-3)
                @test first_step_error(LeanODE.VCABM(), step_size) ≤ step_size^2
            end
        end
        @testset "fixed steps converge at order 1" begin
            coarse, fine = fixed_step_error(LeanODE.VCABM(), 1.0e-2), fixed_step_error(LeanODE.VCABM(), 5.0e-3)
            @test coarse < 1.0e-2
            @test 1.8 < coarse / fine < 2.2
        end
    end

    @testset "accuracy on exponential decay" begin
        problem = LeanODE.ODEProblem(decay!, [1.0, 2.0], (0.0, 1.0), 1.0)
        for alg in ALGORITHMS
            sol = LeanODE.solve(problem, alg; reltol = 1e-10, abstol = 1e-12)
            @test sol.retcode == ReturnCode.Success
            @test sol.t[1] == 0.0 && sol.t[end] == 1.0
            @test sol.u[end] ≈ exp(-1) .* [1.0, 2.0] rtol = 1e-7
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

    @testset "the finiteness check equals any(!isfinite, u)" begin
        finite = [0.0, -0.0, 1.0, -1.0, floatmax(), -floatmax(), floatmin(), 5.0e-324]
        for n in (0, 1, 2, 3, 7, 8, 9, 16, 17, 33), bad in (1.0, Inf, -Inf, NaN), position in unique((1, max(n, 1)))
            x = [finite[mod1(i, length(finite))] for i in 1:n]
            n > 0 && (x[position] = bad)
            @test LeanODE.infinite_or_giant(x) == any(!isfinite, x)
            for z in (complex.(x, 1.0), complex.(1.0, x), reshape(complex.(x, x), :, 1))
                @test LeanODE.infinite_or_giant(z) == any(!isfinite, z)
            end
        end
    end

    @testset "a tolerance per component is refused" begin
        problem = LeanODE.ODEProblem(decay!, [1.0, 2.0], (0.0, 1.0), 1.0)
        @test_throws ArgumentError LeanODE.solve(problem, LeanODE.Tsit5(); abstol = [1.0e-6, 1.0e-8])
        @test_throws ArgumentError LeanODE.solve(problem, LeanODE.Tsit5(); reltol = [1.0e-6, 1.0e-8])
    end

    @testset "fixed steps need dt" begin
        problem = LeanODE.ODEProblem(decay!, [1.0], (0.0, 1.0), 1.0)
        @test_throws ArgumentError LeanODE.solve(problem, LeanODE.RK4(); adaptive = false)
    end
end
