# LeanODE's methods are OrdinaryDiffEq's, at the versions pinned in
# test/Project.toml: the same coefficients, error estimates and step-size
# control. The suite solves the same problems with both and checks that the
# results agree to rounding error where the steps are the same, and within the
# tolerance where step-size control lets rounding differences grow
# (test_agreement.jl).

using Test
using LeanODE
import OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, OrdinaryDiffEqLowOrderRK,
    OrdinaryDiffEqAdamsBashforthMoulton, DiffEqBase, SciMLBase, FastPower
using Logging: NullLogger, with_logger

# The bounds in test_agreement.jl were measured against these versions, and
# the code was written from them, so the suite stops if the environment drifted.
const REFERENCE = [
    OrdinaryDiffEqCore => v"3.33.1",
    OrdinaryDiffEqTsit5 => v"1.12.0",
    OrdinaryDiffEqLowOrderRK => v"1.13.0",
    OrdinaryDiffEqAdamsBashforthMoulton => v"1.11.0",
    DiffEqBase => v"6.218.0",
    SciMLBase => v"2.155.2",
    FastPower => v"1.5.1",
]

@testset "LeanODE" begin
    @testset "reference versions" begin
        for (mod, version) in REFERENCE
            @test pkgversion(mod) == version
        end
    end
    include("problems.jl")
    include("compare.jl")
    include("cases.jl")
    include("test_agreement.jl")
    include("test_behaviour.jl")
    include("test_extension.jl")
end
