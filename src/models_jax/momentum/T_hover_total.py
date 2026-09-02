import jax.numpy as jnp

def total_thrust_required_hover(MTOM, g):
    return 1.25 *MTOM * g
