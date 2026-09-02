import jax.numpy as jnp

def lift_coefficient_to_lift(CL, rho, V, S):
    return 0.5 * rho * V ** 2 * S * CL


def cl_required_cruise(MTOM, g, rho, V_cruise, b, c):
    """Required cruise CL from force balance."""
    s_ref = jnp.maximum(b * c, 1e-9)
    v_safe = jnp.maximum(V_cruise, 1e-6)
    return (2.0 * MTOM * g) / (rho * v_safe ** 2 * s_ref)


def cl_required_climb(MTOM, g, rho, V_climb, b, c, gamma_deg):
    """Required climb CL from force balance with climb angle gamma."""
    s_ref = jnp.maximum(b * c, 1e-9)
    v_safe = jnp.maximum(V_climb, 1e-6)
    gamma_rad = jnp.deg2rad(gamma_deg)
    return (2.0 * MTOM * g * jnp.cos(gamma_rad)) / (rho * v_safe ** 2 * s_ref)
