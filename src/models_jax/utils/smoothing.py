import jax.numpy as jnp


def _softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def soft_floor(x, floor, k=80.0):
    """Smooth approximation of max(x, floor)."""
    z = k * (x - floor)
    return floor + _softplus(z) / k


def soft_cap(x, cap, k=80.0):
    """Smooth approximation of min(x, cap)."""
    z = k * (cap - x)
    return cap - _softplus(z) / k
