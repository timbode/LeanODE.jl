# LeanODE

LeanODE solves ordinary differential equations with methods copied from
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). For the same
problem and options, it returns the same results as OrdinaryDiffEq, bit for bit.
It has no dependencies outside Julia's standard library.

## Identical results and compilation

The results are identical when both packages are compiled the same way. Julia
compiles a function either just in time, when it is first called, or ahead of
time, into a package image or a system image, and the two round some
multiply-adds differently.

In an ordinary session each package runs the code it precompiled and compiles
the rest just in time. OrdinaryDiffEq precompiles `Tsit5` for
`Vector{Float64}` states, and LeanODE precompiles the same, so the results are
identical there as well.

The tests compare the packages in all three ways: everything compiled ahead
of time, an ordinary session, and everything compiled just in time.
