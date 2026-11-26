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

## Speed

LeanODE loads in about 10 ms. OrdinaryDiffEqTsit5 and
OrdinaryDiffEqAdamsBashforthMoulton each load about a hundred packages and
take about half a second. Loading and the first solve of the Lorenz system
take together:

| Method | LeanODE | OrdinaryDiffEq |
|---|---|---|
| `Tsit5` | 0.21 s | 0.63 s |
| `VCABM` | 0.18 s | 1.6 s |

The first solve alone takes longer with LeanODE's `Tsit5`: 0.20 s against
0.08 s. LeanODE compiles its solver for each right-hand side. OrdinaryDiffEq
calls the right-hand side through a wrapper, so that the solver it
precompiled serves every right-hand side of a `Vector{Float64}` state. A
program that solves with many different right-hand sides pays LeanODE's
compilation for each of them.

Once compiled, LeanODE takes less time per step. As a fraction of
OrdinaryDiffEq's time per step:

| Method | Saving every step | Saving only the end |
|---|---|---|
| `Tsit5` | 0.48–0.73 | 0.69–0.86 |
| `RK4` | 0.51–0.57 | 0.61–0.69 |
| `BS3` | 0.53–0.68 | 0.68–0.75 |
| `VCABM3` | 0.60–0.72 | 0.64–0.97 |
| `VCABM5` | 0.53–0.74 | 0.56–0.90 |
| `VCABM` | 0.58–0.61 | 0.56–0.65 |

Each range covers three problems at the default tolerances, all on
`t ∈ [0, 10]`: the Lorenz system with 3 components, and independent harmonic
oscillators with 100 and with 10 000 components. Both packages take the same
number of steps, give or take one, so the fractions are also those of the
solve times. Saving every step costs OrdinaryDiffEq more time than LeanODE,
and the fractions are smaller there.

The numbers come from an Apple M4 with Julia 1.12.7 and the OrdinaryDiffEq
versions the tests pin. `benchmark/startup.jl` and `benchmark/per_step.jl`
measure them.

## Agreement with OrdinaryDiffEq

For the same problem and options, LeanODE's results agree with OrdinaryDiffEq's
to rounding error, not bit for bit. LeanODE computes each stage and the error
estimate in fewer passes over the state, and that changes the order of the
floating-point operations. The Adams methods keep the operations of every
component and differ only in the error norm.

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

`VCABM` differs on purpose at order 1. In OrdinaryDiffEqAdamsBashforthMoulton
up to version 2.2.1, a step at order 1 advances by twice the slope. Every
solve starts at order 1, and without step-size control the order stays 1.
LeanODE has the fix of
[SciML/OrdinaryDiffEq.jl#5056](https://github.com/SciML/OrdinaryDiffEq.jl/pull/5056),
and the tests compare its `VCABM` with OrdinaryDiffEq's step with the same
fix.
