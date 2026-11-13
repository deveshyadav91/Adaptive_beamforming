import numpy as np

def ula_positions(n_elements: int, spacing_lambda: float) -> np.ndarray:
    if n_elements < 2 or spacing_lambda <= 0:
        raise ValueError("Need at least 2 elements and positive spacing.")
    x = (np.arange(n_elements) - (n_elements - 1) / 2) * spacing_lambda
    return np.column_stack((x, np.zeros(n_elements), np.zeros(n_elements)))

def upa_positions(nx: int, ny: int, spacing_lambda: float):
    if min(nx, ny) < 1 or spacing_lambda <= 0:
        raise ValueError("Dimensions and spacing must be positive.")
    x = (np.arange(nx) - (nx - 1) / 2) * spacing_lambda
    y = (np.arange(ny) - (ny - 1) / 2) * spacing_lambda
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny)))

def direction_vector(theta_deg: float, phi_deg: float = 0.0):
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])

def steering_vector(positions_lambda, theta_deg, phi_deg=0.0):
    # Positive phase convention for a wave arriving from direction u.
    u = direction_vector(theta_deg, phi_deg)
    return np.exp(1j * 2*np.pi * (np.asarray(positions_lambda) @ u))

def array_factor(positions_lambda, weights, theta_deg, phi_deg=0.0):
    a = steering_vector(positions_lambda, theta_deg, phi_deg)
    return np.vdot(np.asarray(weights), a)

def db20(x, floor_db=-120):
    return np.maximum(20*np.log10(np.maximum(np.abs(x), 1e-15)), floor_db)


def steering_matrix(positions_lambda, theta_deg, phi_deg=0.0):
    """Vectorised steering vectors, shape (n_elements, n_angles).

    Column k equals steering_vector(positions, theta[k], phi[k]).
    """
    th, ph = np.broadcast_arrays(np.atleast_1d(np.asarray(theta_deg, float)),
                                 np.atleast_1d(np.asarray(phi_deg, float)))
    t, p = np.deg2rad(th), np.deg2rad(ph)
    u = np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)))
    return np.exp(1j * 2 * np.pi * (np.asarray(positions_lambda) @ u))
