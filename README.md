# LeanODE

LeanODE solves ordinary differential equations with methods copied from
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). For the same
problem and options, it returns the same results as OrdinaryDiffEq, bit for bit.
It has no dependencies outside Julia's standard library.

## Identical results and compilation

The results are identical when both packages are compiled the same way. Julia
compiles a function either just in time, when it is first called, or ahead of
time, into a package image or a system image, and the two round some
multiply-adds differently. The tests compare the packages in both ways:
everything compiled just in time, and everything compiled ahead of time.

In an ordinary session the two are mixed. OrdinaryDiffEq precompiles `Tsit5`
for `Vector{Float64}` states, while LeanODE's code is compiled just in time,
so that case can differ in the last bits of the solution. The tolerances are
met either way.

Work in progress.
