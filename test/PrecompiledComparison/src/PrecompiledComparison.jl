"""
    PrecompiledComparison

A test fixture: the comparisons of `test/cases.jl`, compiled ahead of time.

Julia compiles a function either just in time, when it is first called, or
ahead of time, into the package image written when a package is precompiled.
The two can round a multiply-add differently, so LeanODE and OrdinaryDiffEq
are only expected to agree when both are compiled the same way. The other
tests compile both just in time. This package runs every comparison while it
is precompiled, so that its package image holds ahead-of-time code for both
sides; calling [`run_all`](@ref) afterwards runs that code.
"""
module PrecompiledComparison

using LeanODE
import OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, OrdinaryDiffEqLowOrderRK, SciMLBase
using Logging: NullLogger, with_logger

const TESTDIR = joinpath(@__DIR__, "..", "..")
include(joinpath(TESTDIR, "problems.jl"))
include(joinpath(TESTDIR, "compare.jl"))
include(joinpath(TESTDIR, "cases.jl"))

"""
    run_all()

Run every comparison of `comparison_cases()`. Returns one named tuple per
case: its `group`, `label` and method name `alg`, and the fields of
[`compare`](@ref).
"""
function run_all()
    return [
        (; case.group, case.label, alg = nameof(typeof(case.alg)), compare(case)...)
            for case in comparison_cases()
    ]
end

# Run the comparisons once while precompiling, so that the code they reach is
# compiled into the package image. `Pkg.test` forces bounds checks, which
# changes the compiled loops; that variant of the image is never used for the
# comparison (runtests.jl starts its own processes), so it skips the work.
if ccall(:jl_generating_output, Cint, ()) == 1 && Base.JLOptions().check_bounds != 1
    run_all()
end

end # module PrecompiledComparison
