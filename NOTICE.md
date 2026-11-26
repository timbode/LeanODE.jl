# Code from other packages

LeanODE's methods are taken from OrdinaryDiffEq.jl and its base packages.
Each entry names the source file and the version it was taken from. Their
licences are in `LICENSE.md`. The step loop, the step-size control and the
saving are copied. In the methods themselves, the coefficients and formulas
are OrdinaryDiffEq's and the loops over the state are LeanODE's own
(`src/kernels.jl`, `src/hermite.jl`, `src/adams_utils.jl`).

| LeanODE file | Copied from |
|---|---|
| `src/vendor/FastPower.jl` | FastPower.jl 1.5.1, `src/FastPower.jl` |
| `src/defaults.jl` | DiffEqBase.jl 6.218.0, `src/common_defaults.jl`, `src/utils.jl` |
| `src/kernels.jl` | the error scaling of DiffEqBase.jl 6.218.0, `src/calculate_residuals.jl` |
| `src/misc_utils.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/misc_utils.jl`; RecursiveArrayTools.jl 3.54.0, `src/utils.jl` |
| `src/integrator.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/integrators/type.jl` (reduced) |
| `src/initdt.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/initdt.jl` |
| `src/controller.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/integrators/controllers.jl`, `src/alg_utils.jl` |
| `src/saving.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/integrators/integrator_utils.jl`, `src/dense/generic_dense.jl`; RecursiveArrayTools.jl 3.54.0, `src/utils.jl` |
| `src/solve.jl` | OrdinaryDiffEqCore.jl 3.33.1, `src/solve.jl`, `src/integrators/integrator_utils.jl`, `src/integrators/integrator_interface.jl`; SciMLBase.jl 2.155.2, `src/integrator_interface.jl` |
| `src/tsit5_tableaus.jl` | OrdinaryDiffEqTsit5.jl 1.12.0, `src/tsit_tableaus.jl` (unchanged, docstrings added) |
| `src/tsit5.jl` | OrdinaryDiffEqTsit5.jl 1.12.0, `src/tsit_caches.jl`, `src/tsit_perform_step.jl`, `src/interpolants.jl` |
| `src/rk4.jl` | OrdinaryDiffEqLowOrderRK.jl 1.13.0, `src/low_order_rk_caches.jl`, `src/fixed_timestep_perform_step.jl` |
| `src/bs3.jl` | OrdinaryDiffEqLowOrderRK.jl 1.13.0, `src/low_order_rk_tableaus.jl`, `src/low_order_rk_caches.jl`, `src/low_order_rk_perform_step.jl` |
| `src/adams_caches.jl` | OrdinaryDiffEqAdamsBashforthMoulton.jl 1.11.0, `src/algorithms.jl`, `src/alg_utils.jl`, `src/adams_bashforth_moulton_caches.jl` |
| `src/adams_utils.jl` | OrdinaryDiffEqAdamsBashforthMoulton.jl 1.11.0, `src/adams_utils.jl` |
| `src/adams_perform_step.jl` | OrdinaryDiffEqAdamsBashforthMoulton.jl 1.11.0, `src/adams_bashforth_moulton_perform_step.jl` |
| `src/hermite.jl` | the interpolant of OrdinaryDiffEqCore.jl 3.33.1, `src/dense/generic_dense.jl` |
| `test/vcabm_reference.jl` | OrdinaryDiffEqAdamsBashforthMoulton.jl 1.11.0, `src/adams_bashforth_moulton_perform_step.jl` (one method, with the order-1 fix of [SciML/OrdinaryDiffEq.jl#5056](https://github.com/SciML/OrdinaryDiffEq.jl/pull/5056)) |
