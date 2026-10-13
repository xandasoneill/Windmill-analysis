"""
Sweep of the wind direction (yaw beta x vertical angle gamma) on the main shaft.

For every (beta, gamma) combination a MainShaft is solved and the maximum resultant bending moment
and maximum resultant shear force are stored. The wind force magnitude is kept constant for all
directions (F_wind_total of MainShaft), so the map shows the effect of the direction only.

Results are saved to separate CSV files based on F_wind_total and Torque, so sweeps with different
wind magnitudes don't overwrite each other.
"""
import csv
import os
import time

import numpy as np

from main_shaft import MainShaft

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "windsweep_results")


def _ensure_results_dir():
    """Create the windsweep_results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _results_filename(F_wind_total=9101.67, Torque=2463.51):
    """Generate a filename based on wind force and torque parameters."""
    _ensure_results_dir()
    return os.path.join(RESULTS_DIR, f"wind_sweep_F{F_wind_total:.0f}_T{Torque:.0f}.csv")


def sweep_wind_directions(beta_values, gamma_values, **shaft_kwargs):
    """
    Solve the main shaft for every (beta, gamma) combination.

    beta_values, gamma_values : 1D sequences of angles [deg]
    shaft_kwargs              : any MainShaft parameter (F_wind_total, Torque, l1, ...)

    Returns a dict of arrays; the 2D ones have shape (len(gamma_values), len(beta_values)),
    i.e. results[j, i] belongs to (gamma_values[j], beta_values[i]):
        beta, gamma : the 1D angle axes [deg]
        M_max       : maximum resultant bending moment [N·m]
        z_max_M     : position of M_max along the shaft [m]
        V_max       : maximum resultant shear force [N]
    """
    beta_values = np.asarray(beta_values, dtype=float)
    gamma_values = np.asarray(gamma_values, dtype=float)
    shape = (len(gamma_values), len(beta_values))
    M_max, z_max_M, V_max = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    for j, gamma in enumerate(gamma_values):
        for i, beta in enumerate(beta_values):
            shaft = MainShaft(beta_deg=beta, gamma_deg=gamma, **shaft_kwargs)
            M_max[j, i] = shaft.M_max
            z_max_M[j, i] = shaft.z_max_M
            V_max[j, i] = np.max(shaft.V_res)

    return {"beta": beta_values, "gamma": gamma_values,
            "M_max": M_max, "z_max_M": z_max_M, "V_max": V_max}


def save_results(results, path=None, F_wind_total=9101.67, Torque=2463.51):
    """
    Save the sweep as a CSV with one row per (beta, gamma) combination.

    If path is None, it is generated from F_wind_total and Torque, so different wind cases
    are saved to different files (e.g. wind_sweep_F9102_T2464.csv).
    The filename encodes the F_wind_total and Torque values for easy reference.
    """
    if path is None:
        path = _results_filename(F_wind_total, Torque)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta_deg", "gamma_deg", "M_max_Nm", "z_max_M_m", "V_max_N"])
        for j, gamma in enumerate(results["gamma"]):
            for i, beta in enumerate(results["beta"]):
                writer.writerow([beta, gamma, results["M_max"][j, i],
                                 results["z_max_M"][j, i], results["V_max"][j, i]])
    return path


def load_results(path=None, F_wind_total=9101.67, Torque=2463.51):
    """
    Read back a CSV written by save_results, in the same format returned by sweep_wind_directions.

    If path is None, it is generated from F_wind_total and Torque (same as save_results).
    Returns the dict, or None if the file does not exist.
    """
    if path is None:
        path = _results_filename(F_wind_total, Torque)

    if not os.path.exists(path):
        return None

    data = np.genfromtxt(path, delimiter=",", names=True)
    beta = np.unique(data["beta_deg"])
    gamma = np.unique(data["gamma_deg"])
    shape = (len(gamma), len(beta))
    j = np.searchsorted(gamma, data["gamma_deg"])
    i = np.searchsorted(beta, data["beta_deg"])
    out = {"beta": beta, "gamma": gamma}
    for key, column in (("M_max", "M_max_Nm"), ("z_max_M", "z_max_M_m"), ("V_max", "V_max_N")):
        out[key] = np.zeros(shape)
        out[key][j, i] = data[column]
    return out


def print_summary(results):
    """Print the worst wind direction for bending moment and shear force."""
    for key, label, unit in (("M_max", "Maximum bending moment", "kN·m"), ("V_max", "Maximum shear force", "kN")):
        j, i = np.unravel_index(np.argmax(results[key]), results[key].shape)
        print(f"{label}: {results[key][j, i]/1000:.3f} {unit} at beta = {results['beta'][i]:.0f}°, "
              f"gamma = {results['gamma'][j]:.0f}°")


if __name__ == "__main__":
    from visuals import plot_direction_map

    # beta: 0 = wind aligned with the shaft projection (frontal), ±90 = side, ±180 = from behind
    betas = np.arange(-180, 180 + 1, 5)
    gammas = np.arange(-45, 45 + 1, 5)

    F_wind_total = 9101.67
    Torque = 2463.51

    # Try to load existing results for these parameters
    results = load_results(F_wind_total=F_wind_total, Torque=Torque)

    if results is None:
        print(f"Computing sweep (F_wind_total={F_wind_total} N, Torque={Torque} N·m)...")
        start = time.time()
        results = sweep_wind_directions(betas, gammas, F_wind_total=F_wind_total, Torque=Torque)
        elapsed = time.time() - start
        print(f"{len(betas) * len(gammas)} combinations solved in {elapsed:.1f} s")
        path = save_results(results, F_wind_total=F_wind_total, Torque=Torque)
        print(f"Results saved to {path}")
    else:
        print(f"Loaded existing results from {_results_filename(F_wind_total, Torque)}")

    print_summary(results)

    # Save maps to windsweep_results folder
    plot_direction_map(results["beta"], results["gamma"], results["M_max"],
                       cbar_label="Maximum bending moment M_max [kN·m]",
                       title=f"Main shaft: M_max vs wind direction (F={F_wind_total:.0f} N)",
                       save_path=os.path.join(RESULTS_DIR, f"map_Mmax_F{F_wind_total:.0f}.png"),
                       show=False)
    plot_direction_map(results["beta"], results["gamma"], results["V_max"],
                       cbar_label="Maximum shear force V_max [kN]",
                       title=f"Main shaft: V_max vs wind direction (F={F_wind_total:.0f} N)",
                       save_path=os.path.join(RESULTS_DIR, f"map_Vmax_F{F_wind_total:.0f}.png"))
