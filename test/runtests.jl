# LeanODE's results must equal OrdinaryDiffEq's bit for bit, at the versions
# pinned in test/Project.toml.
#
# Both sides must be compiled the same way for that comparison to mean anything.
# OrdinaryDiffEq precompiles some of its kernels (for `Vector{Float64}` states)
# into its package image, and that native code fuses multiply-adds differently
# from the same source compiled just in time. So the suite runs with package
# images disabled; if they are enabled, it starts itself again without them.

if Base.JLOptions().use_pkgimages != 0
    cmd = `$(Base.julia_cmd()) --pkgimages=no --startup-file=no --project=$(Base.active_project()) $(@__FILE__)`
    exit(run(ignorestatus(cmd)).exitcode)
end

using Test
using LeanODE
import OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, OrdinaryDiffEqLowOrderRK,
    OrdinaryDiffEqAdamsBashforthMoulton, DiffEqBase, SciMLBase,
    FastPower, MuladdMacro, FastBroadcast
using Logging: NullLogger, with_logger

# A comparison against any other version proves nothing, so the suite stops
# if the environment drifted.
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

include("problems.jl")
include("compare.jl")

@testset "LeanODE" begin
    @testset "reference versions" begin
        for (mod, version) in REFERENCE
            @test pkgversion(mod) == version
        end
        @test VERSION == v"1.12.7"
    end
    include("test_identical.jl")
    include("test_behaviour.jl")
    include("test_extension.jl")
end
