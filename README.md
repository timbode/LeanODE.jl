# LeanODE

LeanODE solves ordinary differential equations with methods taken from
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl): the same
coefficients, error estimates and step-size control. It has no dependencies
outside Julia's standard library.

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

## Agreement with OrdinaryDiffEq

For the same problem and options, LeanODE's results agree with OrdinaryDiffEq's
to rounding error, not bit for bit. LeanODE computes each stage and the error
estimate in fewer passes over the state, and that changes the order of the
floating-point operations.

- One step from the same state gives the same new state to about `1e-14` of
  its largest component.
- With a fixed step size both packages take the same steps, and the solutions
  differ by accumulated rounding error.
- With step-size control, rounding differences in the error estimate change
  the step sizes slightly. The two solutions then differ by a fraction of the
  tolerance. In a chaotic system this difference grows with time, like any
  other perturbation.

The tests bound all three against OrdinaryDiffEq at the versions pinned in
`test/Project.toml`.
