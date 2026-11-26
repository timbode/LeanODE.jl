# Textbook test problems. Each is a named tuple with the right-hand side `f!`,
# the initial state `u0`, the time span and the parameters `p`.

"""Exponential decay `u' = -λu`: the trivial case."""
function decay!(du, u, λ, t)
    @. du = -λ * u
    return nothing
end

"""Lorenz system, `σ = 10, ρ = 28, β = 8/3`: chaotic, so a difference in the last bit grows until it is visible."""
function lorenz!(du, u, (σ, ρ, β), t)
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

"""Van der Pol oscillator `x'' = μ(1 - x²)x' - x`, as a first-order system."""
function van_der_pol!(du, u, μ, t)
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
    return nothing
end

"""Diffusion `u_t = D Δu` on a square grid with zero boundary values, discretised by the five-point stencil; the state is a matrix."""
function diffusion!(du, u, (D, h), t)
    n, m = size(u)
    for j in 1:m, i in 1:n
        up = i > 1 ? u[i - 1, j] : 0.0
        down = i < n ? u[i + 1, j] : 0.0
        left = j > 1 ? u[i, j - 1] : 0.0
        right = j < m ? u[i, j + 1] : 0.0
        du[i, j] = D * (up + down + left + right - 4u[i, j]) / h^2
    end
    return nothing
end

"""Rotation in the complex plane, `u' = iωu`: a complex state whose modulus stays 1."""
function rotation!(du, u, ω, t)
    @. du = im * ω * u
    return nothing
end

"""Reaction–diffusion `u_t = κ u_xx - u³` on a line with zero boundary values, by central differences: a vector long enough to fill the vectorised part of every loop and leave a remainder."""
function reaction_diffusion!(du, u, κ, t)
    n = length(u)
    for i in 1:n
        left = i > 1 ? u[i - 1] : 0.0
        right = i < n ? u[i + 1] : 0.0
        du[i] = κ * (left - 2u[i] + right) - u[i]^3
    end
    return nothing
end

"""Blow-up in finite time, `u' = u²`: the solution of `u(0) = 1` is `1 / (1 - t)`, so the step size collapses near `t = 1`."""
function blowup!(du, u, p, t)
    @. du = u^2
    return nothing
end

"""A right-hand side that becomes NaN after `t = t_nan`, to test the `Unstable` return code."""
function nan_after!(du, u, t_nan, t)
    @. du = t > t_nan ? NaN : -u
    return nothing
end

const grid_u0 = [exp(-20((i / 9 - 0.5)^2 + (j / 9 - 0.5)^2)) for i in 1:8, j in 1:8]

const PROBLEMS = (
    decay = (f! = decay!, u0 = [1.0, 2.0, -0.5], tspan = (0.0, 2.0), p = 1.5),
    decay_backwards = (f! = decay!, u0 = [1.0, 2.0], tspan = (1.0, -1.0), p = 1.5),
    lorenz = (f! = lorenz!, u0 = [1.0, 0.0, 0.0], tspan = (0.0, 20.0), p = (10.0, 28.0, 8 / 3)),
    van_der_pol = (f! = van_der_pol!, u0 = [2.0, 0.0], tspan = (0.0, 20.0), p = 1.0),
    van_der_pol_stiff = (f! = van_der_pol!, u0 = [2.0, 0.0], tspan = (0.0, 20.0), p = 20.0),
    diffusion = (f! = diffusion!, u0 = grid_u0, tspan = (0.0, 0.05), p = (1.0, 1 / 9)),
    rotation = (f! = rotation!, u0 = ComplexF64[1, 0.5im, -0.3 + 0.4im], tspan = (0.0, 3.0), p = 2π),
    reaction_diffusion = (f! = reaction_diffusion!, u0 = [sin(0.37i) for i in 1:37], tspan = (0.0, 3.0), p = 0.8),
)
