import numpy as np

def conventional_weights(a_desired):
    a = np.asarray(a_desired, dtype=complex)
    return a / max(np.vdot(a, a).real, 1e-15)

def sample_covariance(X):
    X = np.asarray(X, dtype=complex)
    return (X @ X.conj().T) / X.shape[1]

def mvdr_weights(X, a_desired, diagonal_loading=1e-3):
    R = sample_covariance(X)
    scale = np.trace(R).real / R.shape[0]
    R_loaded = R + diagonal_loading * max(scale, 1e-12) * np.eye(R.shape[0])
    z = np.linalg.solve(R_loaded, np.asarray(a_desired, complex))
    denom = np.vdot(a_desired, z)
    if abs(denom) < 1e-15:
        raise np.linalg.LinAlgError("MVDR normalization denominator is too small.")
    return z / denom

def lms_train(X, desired_reference, mu=1e-3, epochs=3, initial_weights=None):
    X = np.asarray(X, complex)
    d = np.asarray(desired_reference, complex).reshape(-1)
    n = X.shape[0]
    w = np.zeros(n, complex) if initial_weights is None else np.asarray(initial_weights, complex).copy()
    errors = []
    # Normalized LMS-style safe step scaling is intentionally not used:
    # this is conventional complex LMS with user-controlled mu.
    for _ in range(epochs):
        e2 = []
        for k in range(X.shape[1]):
            x = X[:, k]
            e = d[k] - np.vdot(w, x)
            w += mu * x * np.conj(e)
            e2.append(abs(e)**2)
        errors.append(float(np.mean(e2)))
    return w, np.asarray(errors)
