import jax.numpy as jnp

def horizontal_climb_speed(V_climb, gamma_deg_climb):
    gamma_rad = jnp.deg2rad(gamma_deg_climb)
    return V_climb * jnp.cos(gamma_rad)
