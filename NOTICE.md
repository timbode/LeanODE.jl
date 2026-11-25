# Copied code

LeanODE's methods are copied from OrdinaryDiffEq.jl and its base packages, so
that LeanODE's results are identical, bit for bit, to theirs. Each entry names
the source file and the version it was copied from. Their licences are in
`LICENSE.md`.

| LeanODE file | Copied from |
|---|---|
| `src/vendor/MuladdMacro.jl` | MuladdMacro.jl 0.2.7, `src/MuladdMacro.jl` |
| `src/vendor/FastPower.jl` | FastPower.jl 1.5.1, `src/FastPower.jl` |
| `src/vendor/FastBroadcast.jl` | FastBroadcast.jl 1.5.1, `src/FastBroadcast.jl`, `ext/FastBroadcastStaticExt.jl`; `indices_do_not_alias` from ArrayInterface.jl 7.30.2 |
| `src/defaults.jl` | DiffEqBase.jl 6.218.0, `src/common_defaults.jl`, `src/calculate_residuals.jl`, `src/utils.jl` |
| `src/misc_utils.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/misc_utils.jl`; RecursiveArrayTools.jl 3.54.0, `src/utils.jl` |
| `src/integrator.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/integrators/type.jl` (reduced) |
| `src/initdt.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/initdt.jl` |
| `src/controller.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/integrators/controllers.jl`, `src/alg_utils.jl` |
| `src/saving.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/integrators/integrator_utils.jl`, `src/dense/generic_dense.jl`; RecursiveArrayTools.jl 3.54.0, `src/utils.jl` |
| `src/solve.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/solve.jl`, `src/integrators/integrator_utils.jl`, `src/integrators/integrator_interface.jl`; SciMLBase.jl 2.155.2, `src/integrator_interface.jl` |
| `src/tsit5_tableaus.jl` | OrdinaryDiffEqTsit5.jl 1.12.0, `src/tsit_tableaus.jl` (unchanged, docstrings added) |
| `src/tsit5.jl` | OrdinaryDiffEqTsit5.jl 1.12.0, `src/tsit_caches.jl`, `src/tsit_perform_step.jl`, `src/interpolants.jl` |
| `src/rk4.jl` | OrdinaryDiffEqLowOrderRK.jl 1.13.0, `src/low_order_rk_caches.jl`, `src/fixed_timestep_perform_step.jl`; OrdinaryDiffEqCore.jl 3.33.1, `src/dense/generic_dense.jl` |
