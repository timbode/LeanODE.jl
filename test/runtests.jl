# LeanODE's results must equal OrdinaryDiffEq's bit for bit, at the versions
# pinned in test/Project.toml.
#
# Both sides must be compiled the same way for that comparison to mean
# anything. Julia compiles a function just in time when it is first called, or
# ahead of time into a package image, and the two round some multiply-adds
# differently: `muladd(c₂, k₂, c₁ * k₁)` becomes `fma(c₂, k₂, c₁ * k₁)` just in
# time and `fma(c₁, k₁, c₂ * k₂)` ahead of time (Julia 1.12.7 on an Apple M4).
# OrdinaryDiffEq precompiles Tsit5 for `Vector{Float64}` states, so in an
# ordinary session that one case runs ahead-of-time code while everything else,
# LeanODE included, is compiled just in time.
#
# The suite therefore runs twice, each time in a process of its own:
#
#   "precompiled": the fixture package test/PrecompiledComparison holds every
#                  comparison compiled ahead of time, for both sides.
#   "jit":         package images are switched off, so both sides are compiled
#                  just in time.
#
# Both processes run with default bounds checking, whatever this one was
# started with (`Pkg.test` forces bounds checks, which changes the loops).

const MODE = get(ENV, "LEANODE_TEST_MODE", "")

if isempty(MODE)
    exitcodes = map(("precompiled" => "yes", "jit" => "no")) do (mode, pkgimages)
        # Julia lists in this file every method it compiles while the process
        # runs; the precompiled mode checks that the solvers are not among them.
        compiled = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --check-bounds=auto --pkgimages=$pkgimages --trace-compile=$compiled --project=$(Base.active_project()) $(@__FILE__)`
        environment = ("LEANODE_TEST_MODE" => mode, "LEANODE_TEST_COMPILED" => compiled)
        run(ignorestatus(addenv(cmd, environment...))).exitcode
    end
    exit(all(iszero, exitcodes) ? 0 : 1)
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

@testset "LeanODE, $MODE" begin
    @testset "reference versions and compile mode" begin
        for (mod, version) in REFERENCE
            @test pkgversion(mod) == version
        end
        @test VERSION == v"1.12.7"
        @test Base.JLOptions().check_bounds == 0
        @test (Base.JLOptions().use_pkgimages != 0) == (MODE == "precompiled")
    end
    if MODE == "precompiled"
        include("test_precompiled.jl")
    else
        include("problems.jl")
        include("compare.jl")
        include("cases.jl")
        include("test_identical.jl")
        include("test_behaviour.jl")
        include("test_extension.jl")
    end
end
