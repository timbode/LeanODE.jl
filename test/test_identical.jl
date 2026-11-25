# LeanODE and OrdinaryDiffEq must give identical solutions in every case of
# cases.jl. The right-hand sides are defined in this process, so each solve is
# compiled now: entirely just in time in the "jit" mode, and around the
# kernels the packages precompiled themselves in the "session" mode.

@testset "identical to OrdinaryDiffEq" begin
    @testset "$(nameof(typeof(alg)))" for alg in ALGORITHMS
        cases = filter(case -> case.alg === alg, comparison_cases())
        rejections = 0
        @testset "$group" for group in unique(case.group for case in cases)
            for case in filter(case -> case.group == group, cases)
                result = compare(case)
                result.same || println("  differs: ", case.label, ": ", result.detail)
                @test result.same
                @test result.retcode_as_expected
                rejections += result.rejections
            end
        end
        @testset "the rejection path is exercised" begin
            @test rejections > 0
        end
    end
end
