# LeanODE and OrdinaryDiffEq, both compiled ahead of time, must give identical
# solutions in every case of cases.jl. The fixture package ran all of them
# while it was precompiled; running them again here uses that compiled code.

using PrecompiledComparison

"""
    solver_methods_compiled_now()

The lines of Julia's compilation log (`--trace-compile`, set up by
runtests.jl) that name a function of LeanODE, of OrdinaryDiffEq or of the
packages below them. Each line is a method this process had to compile just
in time, which the precompiled comparison must not need.
"""
function solver_methods_compiled_now()
    owner = r"typeof\((LeanODE|OrdinaryDiffEq\w*|SciMLBase|DiffEqBase|CommonSolve|FastBroadcast|FastPower|MuladdMacro|RecursiveArrayTools|FunctionWrappers\w*)\."
    return filter(line -> occursin(owner, line), readlines(ENV["LEANODE_TEST_COMPILED"]))
end

@testset "identical to OrdinaryDiffEq, precompiled" begin
    results = PrecompiledComparison.run_all()

    @testset "the solvers ran precompiled code" begin
        compiled_now = solver_methods_compiled_now()
        foreach(println, compiled_now)
        @test isempty(compiled_now)
    end

    @testset "$alg" for alg in unique(result.alg for result in results)
        of_alg = filter(result -> result.alg == alg, results)
        @testset "$group" for group in unique(result.group for result in of_alg)
            for result in filter(result -> result.group == group, of_alg)
                result.same || println("  differs: ", result.label, ": ", result.detail)
                @test result.same
                @test result.retcode_as_expected
            end
        end
        @testset "the rejection path is exercised" begin
            @test sum(result.rejections for result in of_alg) > 0
        end
    end
end
