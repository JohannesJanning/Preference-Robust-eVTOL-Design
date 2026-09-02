import jax.numpy as jnp

def total_thrust_required_climb(D_climb, MTOM, g, gamma_deg_climb):
    gamma_rad = jnp.deg2rad(gamma_deg_climb)
    return D_climb + MTOM * g * jnp.sin(gamma_rad)
