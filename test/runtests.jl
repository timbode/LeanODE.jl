using Test
using LeanODE
import OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, OrdinaryDiffEqLowOrderRK,
       OrdinaryDiffEqAdamsBashforthMoulton, DiffEqBase, SciMLBase,
       FastPower, MuladdMacro, FastBroadcast

# LeanODE must reproduce exactly these versions. A comparison against any other
# version proves nothing, so the suite stops if the environment drifted.
const REFERENCE = [
    OrdinaryDiffEqCore => v"3.33.1",
    OrdinaryDiffEqTsit5 => v"1.12.0",
    OrdinaryDiffEqLowOrderRK => v"1.13.0",
    OrdinaryDiffEqAdamsBashforthMoulton => v"1.11.0",
    DiffEqBase => v"6.218.0",
    SciMLBase => v"2.155.2",
    FastPower => v"1.5.1",
    MuladdMacro => v"0.2.7",
    FastBroadcast => v"1.5.1",
]

@testset "LeanODE" begin
    @testset "reference versions" begin
        for (mod, version) in REFERENCE
            @test pkgversion(mod) == version
        end
        @test VERSION == v"1.12.7"
    end
end
