import jax.numpy as jnp

def gamma_from_roc(roc: float, v_climb: float) -> float:
    """Compute climb angle gamma in degrees from ROC and climb speed."""
    v_safe = jnp.maximum(v_climb, 1e-6)
    ratio = jnp.clip(roc / v_safe, -0.999999, 0.999999)
    return jnp.rad2deg(jnp.arcsin(ratio))


def roc_from_gamma(gamma_deg: float, v_climb: float) -> float:
    """Compute ROC from climb angle gamma in degrees and climb speed."""
    gamma_rad = jnp.deg2rad(gamma_deg)
    return jnp.sin(gamma_rad) * v_climb


def roc_calculation(gamma_deg_climb: float, v_climb: float) -> float:
    """Backward-compatible alias for ROC from gamma.

    Parameters:
        gamma_deg_climb: climb angle in degrees
        v_climb: climb speed (m/s)
    """
    return roc_from_gamma(gamma_deg_climb, v_climb)
