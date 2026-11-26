# LeanODE implements OrdinaryDiffEq's methods with the same coefficients, error
# estimates and step-size control, but its loops over the state do the
# arithmetic in a different order. Results therefore agree with OrdinaryDiffEq
# to rounding error, and these tests bound the difference at four levels.
#
# Each bound leaves a factor of at least six over the largest difference seen
# on an Apple M4 with Julia 1.12.7; the measured values are given with each.

"""
    estimate_noise(reference_state, abstol, reltol)

The size of rounding error in a scaled error estimate: one unit in the last
place of the largest component, divided by that component's tolerance.
"""
function estimate_noise(reference_state, abstol, reltol)
    scale = largest_magnitude((reference_state,))
    return eps(scale) / (abstol + reltol * scale)
end

# One step from the same state with the same step size. The new states differ
# only by rounding (largest seen: 7e-15 of the largest component). The error
# estimate is a small difference of large terms, so its rounding error is
# given in units of `estimate_noise` (largest seen: 26). The proposed next
# step size follows from the error estimate through a power below 1, so it
# cannot differ more, relatively, than the estimate does.
#
# Only a Runge–Kutta method can be started anywhere with the step size it had
# there. An Adams method restarts at low order and rejects that step size.
@testset "one step from the same state" begin
    @testset "$(nameof(typeof(alg)))" for alg in RUNGE_KUTTA_METHODS
        @testset "$name, $tolerance" for (name, problem) in pairs(PROBLEMS), (tolerance, keywords) in pairs(TOLERANCES)
            abstol = get(keywords, :abstol, 1.0e-6)
            reltol = get(keywords, :reltol, 1.0e-3)
            for point in states_along(problem, alg, 12; keywords...)
                lean, reference = one_step(problem, alg, point; keywords...)
                @test lean.t == reference.t
                @test relative_distance((lean.u,), (reference.u,)) ≤ 1.0e-13
                estimate_difference = abs(lean.error_estimate - reference.error_estimate)
                @test estimate_difference ≤ 300 * estimate_noise(reference.u, abstol, reltol)
                @test abs(lean.next_dt / reference.next_dt - 1) ≤ 1.0e-6 + estimate_difference / reference.error_estimate
            end
        end
    end
end

# The first steps of a solve, compared after every step, with error control
# and with the step size held at `fixed_dt` by `dtmax`. This is the sharp
# test of the Adams methods: their step uses the history of the solve, and
# the variable-order method chooses its order from error estimates (it rises
# to order 12 here). Times, orders and rejections must be equal, and states
# and error estimates within rounding as above (largest seen: 1e-15 and 16).
# The Adams methods' states are even equal bit for bit, because their sweeps
# keep upstream's operations per component. The equality matters at high
# order, where the error estimates of neighbouring orders consist of rounding
# error only and a different last bit of the state can change the order.
@testset "the first steps, one at a time" begin
    @testset "$(nameof(typeof(alg)))" for alg in ALGORITHMS
        @testset "$name" for (name, problem) in pairs(PROBLEMS)
            (; fixed_dt, tspan) = problem
            count = min(60, floor(Int, abs(tspan[2] - tspan[1]) / fixed_dt) - 2)
            steps = with_logger(NullLogger()) do
                first_steps(problem, alg, count; dt = fixed_dt, dtmax = fixed_dt)
            end
            for (lean, reference) in steps
                @test lean.t == reference.t
                @test lean.order == reference.order
                @test lean.rejections == reference.rejections
                @test relative_distance((lean.u,), (reference.u,)) ≤ 1.0e-13
                estimate_difference = abs(lean.error_estimate - reference.error_estimate)
                @test estimate_difference ≤ 300 * estimate_noise(reference.u, 1.0e-6, 1.0e-3)
                @test abs(lean.next_dt / reference.next_dt - 1) ≤ 1.0e-6 + estimate_difference / reference.error_estimate
            end
        end
    end
end

# Without step-size control both take the same steps, so the saved times and
# the work counters are identical, and the states differ by accumulated
# rounding (largest seen: 1e-11 of the largest component, for the stiff Van der
# Pol oscillator over 10 000 steps). With `saveat` this also compares the
# interpolants.
@testset "fixed steps" begin
    @testset "$(nameof(typeof(alg)))" for alg in ALGORITHMS
        @testset "$name, $saving" for (name, problem) in pairs(PROBLEMS), (saving, keywords) in pairs(saving_options(problem.tspan))
            lean, reference = both_solutions(problem, alg; adaptive = false, dt = problem.fixed_dt, keywords...)
            @test Symbol(reference.retcode) == :Success
            @test Symbol(lean.retcode) == :Success
            @test isequal(lean.t, reference.t)
            @test same_counts(lean, reference)
            @test relative_distance(lean.u, reference.u) ≤ 1.0e-9
        end
    end
end

# With step-size control, rounding differences in the error estimate change
# the step sizes slightly, and the two solutions drift apart by a fraction of
# the tolerance (largest seen: 1.6 tolerances, for the Lorenz system at the
# default tolerances). Every case must end with OrdinaryDiffEq's return code.
# A case that reaches the end of the time span must also agree within 10
# tolerances at every saved time the two have in common, save at identical
# times unless it saves after every step, and attempt about as many steps
# (on the M4, exactly as many in all but 12 of 1210 cases).
#
# The stiff Van der Pol oscillator is allowed 100 tolerances. The step size
# of an explicit method is there limited by stability, not accuracy, so it
# oscillates and amplifies rounding differences (largest seen: 9.2
# tolerances, with VCABM3; the step counts differ by up to 4 in 1800).
@testset "adaptive steps" begin
    @testset "$(nameof(typeof(alg)))" for alg in ALGORITHMS
        cases = filter(case -> case.alg === alg, comparison_cases())
        rejections = 0
        @testset "$group" for group in unique(case.group for case in cases)
            for case in filter(case -> case.group == group, cases)
                result = agreement(case)
                result.same_retcode && result.retcode_as_expected && (!result.success || result.distance ≤ result.bound) ||
                    println("  differs: ", case.label, ": ", result)
                @test result.same_retcode
                @test result.retcode_as_expected
                rejections += result.rejections
                result.success || continue
                @test result.distance ≤ result.bound
                @test result.every_step || result.same_times
                @test abs(result.steps - result.reference_steps) ≤ 2 + 0.05 * result.reference_steps
            end
        end
        @testset "the rejection path is exercised" begin
            @test rejections > 0
        end
    end
end
