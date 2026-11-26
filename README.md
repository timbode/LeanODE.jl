# LeanODE

LeanODE solves ordinary differential equations with methods copied from
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). For the same
problem and options, it returns the same results as OrdinaryDiffEq, bit for bit.
It has no dependencies outside Julia's standard library.

## Methods

| Method | Description |
|---|---|
| `Tsit5()` | Tsitouras' Runge–Kutta pair of orders 5 and 4 |
| `RK4()` | the classical Runge–Kutta method of order 4, with step-size control |
| `BS3()` | Bogacki and Shampine's Runge–Kutta pair of orders 3 and 2 |
| `VCABM()` | Adams–Bashforth–Moulton with variable step size and variable order, 1 to 12 |
| `VCABM3()`, `VCABM4()`, `VCABM5()` | Adams–Bashforth–Moulton of fixed order 3, 4 or 5 with variable step size |

The Adams methods evaluate the right-hand side twice per step, whatever
their order.

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
