import numpy as np
from .array import array_factor, db20, steering_vector

def pattern_db(positions_lambda, weights, theta_deg, phi_deg=0.0):
    vals = np.array([abs(array_factor(positions_lambda, weights, t, phi_deg))
                     for t in np.asarray(theta_deg)])
    vals /= max(float(vals.max()), 1e-15)
    return db20(vals)

def hpbw_deg_robust(theta_deg, response_db, peak_angle=None):
    th, p = np.asarray(theta_deg, float), np.asarray(response_db, float)
    peak = int(np.argmax(p)) if peak_angle is None else int(np.argmin(abs(th-peak_angle)))
    level = p[peak] - 3.0103
    left, right = peak, peak
    while left > 0 and p[left] >= level: left -= 1
    while right < len(p)-1 and p[right] >= level: right += 1
    if left == 0 or right == len(p)-1 or p[left] > level or p[right] > level:
        return float("nan")
    def cross(i, j):
        return th[i] + (level-p[i])*(th[j]-th[i])/(p[j]-p[i])
    return float(cross(right-1, right)-cross(left, left+1))

def peak_sidelobe_level_db(theta_deg, response_db, main_lobe_center_deg,
                           exclusion_half_width_deg=8.0):
    th, p = np.asarray(theta_deg), np.asarray(response_db)
    mask = abs(th-main_lobe_center_deg) > exclusion_half_width_deg
    return float(np.max(p[mask])) if np.any(mask) else float("nan")

def null_depth_db(theta_deg, response_db, angle_deg, window_deg=1.0):
    th, p = np.asarray(theta_deg), np.asarray(response_db)
    mask = abs(th-angle_deg) <= window_deg
    return float(np.min(p[mask])) if np.any(mask) else float(p[np.argmin(abs(th-angle_deg))])

def visible_grating_lobes_ula(spacing_lambda, steering_angle_deg, tol=1e-10):
    if spacing_lambda <= 0: raise ValueError("Spacing must be positive.")
    s0 = np.sin(np.deg2rad(steering_angle_deg))
    lo = int(np.ceil((-1-s0)*spacing_lambda-tol))
    hi = int(np.floor((1-s0)*spacing_lambda+tol))
    found = []
    for m in range(lo, hi+1):
        if m == 0: continue
        sm = s0 + m/spacing_lambda
        if abs(sm) <= 1+tol:
            found.append({"order": m, "angle_deg": float(np.rad2deg(np.arcsin(np.clip(sm,-1,1))))})
    return found

def output_sinr_db(w, desired_angle, interferer_angles, desired_power,
                   interference_powers, noise_power, positions_lambda):
    a0 = steering_vector(positions_lambda, desired_angle)
    sig = desired_power * abs(np.vdot(w, a0))**2
    interf = sum(power*abs(np.vdot(w, steering_vector(positions_lambda, angle)))**2
                 for angle, power in zip(interferer_angles, interference_powers))
    noise = noise_power * np.vdot(w,w).real
    return float(10*np.log10(max(sig,1e-15)/max(interf+noise,1e-15)))


# ---------------------------------------------------- robustness metrics ---
def main_lobe_peak_index(theta_deg, response_db, target_deg, search_half_width_deg=6.0):
    """Index of the strongest sample within +/- search window of the target angle."""
    th, p = np.asarray(theta_deg), np.asarray(response_db)
    idx = np.flatnonzero(abs(th - target_deg) <= search_half_width_deg)
    if idx.size == 0:
        return int(np.argmin(abs(th - target_deg)))
    return int(idx[np.argmax(p[idx])])


def first_null_indices(response_db, peak_idx):
    """Walk down both sides of the main lobe to its first local minimum (or array edge)."""
    p = np.asarray(response_db)
    left, right = peak_idx, peak_idx
    while left > 0 and p[left - 1] <= p[left]:
        left -= 1
    while right < len(p) - 1 and p[right + 1] <= p[right]:
        right += 1
    return left, right


def peak_sidelobe_level_null_based(theta_deg, response_db, peak_idx):
    """Highest response outside the main lobe (bounded by its first nulls), in dB
    relative to the main-lobe peak. Unlike a fixed exclusion window this adapts to
    tapering, steering angle and spacing. A grating lobe shows up as ~0 dB."""
    p = np.asarray(response_db, float)
    left, right = first_null_indices(p, peak_idx)
    outside = np.concatenate((p[:left], p[right + 1:]))
    if outside.size == 0:
        return float("nan")
    return float(outside.max() - p[peak_idx])
