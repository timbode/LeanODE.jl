# Every problem, crossed with tolerances and ways of saving, for both methods:
# LeanODE and OrdinaryDiffEq must give identical solutions.

"""Saving options for a time span: every step, the end only, and several kinds of `saveat`."""
function saving_options(tspan)
    t0, tf = tspan
    Δ = tf - t0
    return (
        every_step = (;),
        end_only = (; save_everystep = false),
        saveat_grid = (; saveat = collect(range(t0, tf; length = 11))),
        saveat_interior = (; saveat = t0 .+ Δ .* [0.137, 0.5, 0.9, 0.999]),
        saveat_number = (; saveat = abs(Δ) / 7),
        saveat_without_ends = (; saveat = t0 .+ Δ .* [0.25, 0.75], save_start = false, save_end = false),
    )
end

const TOLERANCES = (
    default = (;),
    loose = (; reltol = 1e-4, abstol = 1e-6),
    tight = (; reltol = 1e-10, abstol = 1e-10),
)

const ALGORITHMS = (LeanODE.Tsit5(), LeanODE.RK4())

"""Test that the two solutions are identical; on failure, print where they differ."""
function test_identical(label, lean, reference)
    same = identical(lean, reference)
    same || println("  differs: ", label, ": ", describe_difference(lean, reference))
    @test same
    return same
end

@testset "identical to OrdinaryDiffEq" begin
    @testset "$(nameof(typeof(alg)))" for alg in ALGORITHMS
        @testset "$name" for (name, problem) in pairs(PROBLEMS)
            for (tol_name, tol) in pairs(TOLERANCES), (save_name, save) in pairs(saving_options(problem.tspan))
                lean, reference = both_solutions(problem, alg; tol..., save...)
                test_identical("$name, $tol_name, $save_name", lean, reference)
            end
        end

        @testset "step-size options" begin
            lorenz = merge(PROBLEMS.lorenz, (; tspan = (0.0, 2.0)))
            for (label, kw) in (
                    ("fixed step", (; adaptive = false, dt = 0.01)),
                    ("initial dt given", (; dt = 1e-3)),
                    ("dtmax", (; dtmax = 0.05)),
                )
                lean, reference = both_solutions(lorenz, alg; kw...)
                test_identical("lorenz, $label", lean, reference)
            end
        end

        @testset "early stops" begin
            lean, reference = both_solutions(PROBLEMS.lorenz, alg; maxiters = 50)
            @test Symbol(reference.retcode) == :MaxIters
            test_identical("lorenz, maxiters = 50", lean, reference)

            nan_problem = (f! = nan_after!, u0 = [1.0, 2.0], tspan = (0.0, 1.0), p = 0.5)
            lean, reference = both_solutions(nan_problem, alg)
            @test Symbol(reference.retcode) == :Unstable
            test_identical("NaN after t = 0.5", lean, reference)
        end

        @testset "the rejection path is exercised" begin
            _, reference = both_solutions(PROBLEMS.van_der_pol_stiff, alg)
            @test reference.stats.nreject > 0
        end
    end
end
