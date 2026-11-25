# The integrator state, reduced from OrdinaryDiffEqCore 3.33.1
# (src/integrators/type.jl, `ODEIntegrator`, and `DEOptions`) to the fields that
# the default path of an explicit method reads or writes.

"""
    Options

Settings of one solve, fixed when it starts (OrdinaryDiffEq's `DEOptions`).

# Fields
- `maxiters`: maximum number of attempted steps.
- `save_everystep`, `save_start`, `save_end`: which states are saved;
  `save_end_user` is the `save_end` the caller passed (`nothing` if none).
- `adaptive`: whether the step size is controlled by the error estimate.
- `abstol`, `reltol`: absolute and relative tolerance.
- `gamma`, `qmax`, `qmin`, `qsteady_max`, `qsteady_min`, `qoldinit`: the
  parameters of the step-size controller (see `controller.jl`).
- `beta1`, `beta2`: the PI controller's exponents.
- `dtmax`, `dtmin`: bounds on the step size.
- `tstops`: times the integration must hit, multiplied by the time direction
  and sorted ascending; only the end of the time span here.
- `saveat`: the remaining times to save at, likewise multiplied by the time
  direction and sorted ascending.
- `saveat_cache`: the `saveat` the caller passed.
"""
struct Options{S}
    maxiters::Int
    save_everystep::Bool
    adaptive::Bool
    abstol::Float64
    reltol::Float64
    gamma::Float64
    qmax::Float64
    qmin::Float64
    qsteady_max::Float64
    qsteady_min::Float64
    qoldinit::Float64
    beta1::Float64
    beta2::Float64
    dtmax::Float64
    dtmin::Float64
    tstops::Vector{Float64}
    saveat::Vector{Float64}
    saveat_cache::S
    save_start::Bool
    save_end::Bool
    save_end_user::Union{Nothing, Bool}
end

"""
    Integrator

The state of a solve between steps (OrdinaryDiffEq's `ODEIntegrator`).

The step from `tprev` to `t` took the state from `uprev` to `u`, with step
size `dt`. `k` holds the stage derivatives the method's interpolant needs;
`fsalfirst` is `f(uprev)` and `fsallast` is `f(u)`, since both methods reuse
the last stage of one step as the first stage of the next.

`ts` and `us` collect the saved times and states; `saveiter` counts them.
`tspan_end` is the end of the time span.
The remaining fields are the step-control state of OrdinaryDiffEq, with the
same names and meanings: `EEst` is the scaled error estimate of the last
step, `qold` and `q11` are the PI controller's memory, `dtpropose` is the
proposed next step size, and the flags record whether the last step was
accepted and whether the next one ends exactly on a stop time.
"""
mutable struct Integrator{uType, F, P, Alg, Cache, S}
    f::F
    p::P
    alg::Alg
    cache::Cache
    opts::Options{S}
    stats::Stats
    tdir::Float64
    u::uType
    uprev::uType
    k::Vector{uType}
    fsalfirst::uType
    fsallast::uType
    t::Float64
    tprev::Float64
    tspan_end::Float64
    dt::Float64
    dtcache::Float64
    dtpropose::Float64
    EEst::Float64
    qold::Float64
    q11::Float64
    iter::Int
    success_iter::Int
    saveiter::Int
    accept_step::Bool
    next_step_tstop::Bool
    tstop_target::Float64
    do_error_check::Bool
    retcode::ReturnCode.T
    ts::Vector{Float64}
    us::Vector{uType}
end
