import numpy as np
import matplotlib

# Qt backend (matplotlib default here) crashes on this machine (GLib/GNOME settings error),
# so use Tk. Falls back to the default backend if Tk is not available.
try:
    matplotlib.use("TkAgg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from wind import Wind

# References (files in resources/):
#   [1] Condaxakis et al., "The Evolution of Windmill Design: From Lasithi Plateau Pumping Windmills
#       to Electricity Production", Energies 2026, 19, 829  (resources/The Evolution of Windmill Design_...pdf)
#   [2] Martin Bates, "Argos Hill Windmill" guide  (resources/Argos Hill Windmill Trust.pdf)

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================

# Geometry
alpha_deg = 8.0                         # Shaft inclination angle [degrees] ref: [2] p.22 (neck bearing raised to 8°)
alpha = np.radians(alpha_deg)
l1 = 0.8                                # Overhang length (from neck bearing to rotor) [m]
L = 4.88                                # Distance between Neck (A) and Tail (B) [m]
d1 = 1.2                                # Distance from Neck (A) to Brake Wheel [m]
r_wheel = 1.0                           # Pitch radius of Brake Wheel [m] (note: [2] p.11 gives brake wheel ~8 ft diameter -> ~1.2 m radius)
phi = np.radians(20.0)                  # Gear pressure angle [degrees]Value come from norms ISO, AGMA e DIN

# Masses & Gravity [kg, m/s²]
m_rotor, m_shaft, m_wheel, g = 3500.0, 1500.0, 1000.0, 9.81
W_rotor, W_shaft, W_wheel = m_rotor * g, m_shaft * g, m_wheel * g

# Wind
F_wind_total = 9101.67                  # Aerodynamic wind force [N] ref: [1] Sec. 3.3, p.22 (extreme case, 40 m/s, 80 RPM)
beta_deg = 10.0                         # Wind yaw misalignment angle [degrees]
gamma_deg = 0.0                         # Wind vertical angle [degrees] (>0 = updraft)

# Transmitted torque(needs to come from the reference to grind corn or from the wind)
Torque = 2463.51                        # [N·m] ref: [1] Sec. 3.3, p.22 (Q = 2463.51 N·m, same extreme case as F_wind_total)

# Material & Safety Factor
S_y = 250e6                             # Yield strength [Pa]
n_safety = 2.5                          # Safety factor
S_allow = S_y / n_safety                # Allowable stress [Pa]

# ==============================================================================
# 2. LOADS IN THE SHAFT FRAME (x lateral, y radial, z axial)
# ==============================================================================

# Wind force components
F_wind_x, F_wind_y, F_wind_z = Wind(F_wind_total, alpha_deg, beta_deg, gamma_deg).components

# Gear reaction forces at z = d1
F_tangential = Torque / r_wheel
F_gear_x = -F_tangential
F_gear_y = F_tangential * np.tan(phi)

# Point load at the rotor (z = -l1): wind + weight
F_rotor_x = F_wind_x
F_rotor_y = F_wind_y - W_rotor * np.cos(alpha)
F_rotor_z = F_wind_z - W_rotor * np.sin(alpha)

# Point load at the brake wheel (z = d1): gear reaction + weight
F_wheel_x = F_gear_x
F_wheel_y = F_gear_y - W_wheel * np.cos(alpha)

# Distributed load of the shaft self-weight [N/m]
w_shaft_y = -(W_shaft / L) * np.cos(alpha)

# ==============================================================================
# 3. BEARING REACTIONS
# ==============================================================================

# Tail bearing (B): axial thrust + radial (moments about A = 0)
R_Bz = F_wind_z + (W_rotor + W_shaft + W_wheel) * np.sin(alpha)
R_Bx = (F_rotor_x * l1 - F_wheel_x * d1) / L
R_By = (F_rotor_y * l1 - w_shaft_y * L**2 / 2.0 - F_wheel_y * d1) / L

# Neck bearing (A): radial (force equilibrium)
R_Ax = -(F_rotor_x + F_wheel_x + R_Bx)
R_Ay = -(F_rotor_y + w_shaft_y * L + F_wheel_y + R_By)

R_A_radial = np.sqrt(R_Ax**2 + R_Ay**2)
R_B_radial = np.sqrt(R_Bx**2 + R_By**2)

print("=== BEARING REACTIONS ===")
print(f"Wind force (x, y, z): ({F_wind_x/1000:.3f}, {F_wind_y/1000:.3f}, {F_wind_z/1000:.3f}) kN")
print(f"Neck Bearing A (radial): R_Ax = {R_Ax/1000:.3f} kN, R_Ay = {R_Ay/1000:.3f} kN -> {R_A_radial/1000:.3f} kN")
print(f"Tail Bearing B (radial): R_Bx = {R_Bx/1000:.3f} kN, R_By = {R_By/1000:.3f} kN -> {R_B_radial/1000:.3f} kN")
print(f"Tail Bearing B (axial):  R_Bz = {R_Bz/1000:.3f} kN")

# ==============================================================================
# 4. SHEAR FORCE & BENDING MOMENT ALONG THE SHAFT
# ==============================================================================

z_vals = np.linspace(-l1, L, 1000)
Vx_vals, Vy_vals = np.zeros_like(z_vals), np.zeros_like(z_vals)
Mx_vals, My_vals = np.zeros_like(z_vals), np.zeros_like(z_vals)

for i, z in enumerate(z_vals):
    if z < 0:
        # Overhang [-l1, 0)
        Vx = F_rotor_x
        Vy = F_rotor_y
        Mx = F_rotor_y * (z + l1)
        My = -F_rotor_x * (z + l1)
    elif z < d1:
        # Between Neck Bearing (A) and Brake Wheel [0, d1)
        Vx = F_rotor_x + R_Ax
        Vy = F_rotor_y + R_Ay + w_shaft_y * z
        Mx = F_rotor_y * (z + l1) + R_Ay * z + w_shaft_y * z**2 / 2.0
        My = -F_rotor_x * (z + l1) - R_Ax * z
    else:
        # Between Brake Wheel and Tail Bearing (B) [d1, L]
        Vx = F_rotor_x + R_Ax + F_wheel_x
        Vy = F_rotor_y + R_Ay + w_shaft_y * z + F_wheel_y
        Mx = F_rotor_y * (z + l1) + R_Ay * z + w_shaft_y * z**2 / 2.0 + F_wheel_y * (z - d1)
        My = -F_rotor_x * (z + l1) - R_Ax * z - F_wheel_x * (z - d1)

    Vx_vals[i], Vy_vals[i] = Vx, Vy
    Mx_vals[i], My_vals[i] = Mx, My

V_res_vals = np.sqrt(Vx_vals**2 + Vy_vals**2)
M_res_vals = np.sqrt(Mx_vals**2 + My_vals**2)

idx_max_M = np.argmax(M_res_vals)
M_max = M_res_vals[idx_max_M]
z_max_M = z_vals[idx_max_M]

print(f"\nMaximum Resultant Bending Moment: M_max = {M_max/1000:.3f} kN·m at z = {z_max_M:.2f} m")

# ==============================================================================
# 5. MINIMUM SHAFT DIAMETER (VON MISES / ASME CRITERION)
# ==============================================================================

# Equivalent moment combining bending and torsion
M_eq = np.sqrt(M_max**2 + 0.75 * Torque**2)

# Solid circular cross-section: d = ((32 * n * M_eq) / (pi * S_y))^(1/3)
d_min = ((32 * n_safety * M_eq) / (np.pi * S_y))**(1/3)

print("\n=== SHAFT DIAMETER SIZING ===")
print(f"Equivalent Moment (M_eq): {M_eq/1000:.3f} kN·m")
print(f"Required Minimum Diameter (d_min): {d_min*1000:.1f} mm ({d_min:.3f} m)")
# Sanity check: [2] p.19 gives the Argos Hill neck bearing as ~12 in (~305 mm) and tail bearing ~6 in (~152 mm)

# ==============================================================================
# 6. PLOTTING DIAGRAMS
# ==============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Main Windshaft Mechanical Analysis', fontsize=14, fontweight='bold')


def mark_supports(ax, labels=False):
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='red', linestyle=':', label='Support A (Neck Bearing)' if labels else None)
    ax.axvline(d1, color='orange', linestyle=':', label='Brake Wheel' if labels else None)
    ax.axvline(L, color='purple', linestyle=':', label='Support B (Tail Bearing)' if labels else None)
    ax.grid(True, linestyle='--', alpha=0.6)


# Shear Force Diagram
mark_supports(ax1, labels=True)
ax1.plot(z_vals, Vx_vals / 1000, color='#2ca02c', linewidth=1.5, label='V_x(z)')
ax1.plot(z_vals, Vy_vals / 1000, color='#ff7f0e', linewidth=1.5, label='V_y(z)')
ax1.plot(z_vals, V_res_vals / 1000, color='#1f77b4', linewidth=2, label='Resultant V(z)')
ax1.set_ylabel('Shear Force V [kN]', fontsize=11, fontweight='bold')
ax1.set_title('Shear Force Diagram (SFD)', fontsize=12)
ax1.legend(loc='upper right')

# Bending Moment Diagram
mark_supports(ax2)
ax2.plot(z_vals, Mx_vals / 1000, color='#2ca02c', linewidth=1.5, label='M_x(z)')
ax2.plot(z_vals, My_vals / 1000, color='#ff7f0e', linewidth=1.5, label='M_y(z)')
ax2.plot(z_vals, M_res_vals / 1000, color='#d62728', linewidth=2, label='Resultant M(z)')
ax2.plot(z_max_M, M_max / 1000, 'ko', markersize=8, label=f'M_max = {M_max/1000:.2f} kN·m')
ax2.set_xlabel('Position along shaft z-axis [m]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Bending Moment M [kN·m]', fontsize=11, fontweight='bold')
ax2.set_title('Bending Moment Diagram (BMD)', fontsize=12)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()
