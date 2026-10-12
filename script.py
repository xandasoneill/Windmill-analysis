import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. INPUT PARAMETERS & VARIABLES (Adjustable)
# ==============================================================================

# Geometry and Shaft Angle
alpha_deg = 8.0                         # Shaft inclination angle [degrees]
alpha = np.radians(alpha_deg)

l1 = 0.8                                # Overhang length (from neck bearing to rotor) [m]
L = 4.88                                # Distance between Neck Bearing (A) and Tail Bearing (B) [m]
d1 = 1.2                                # Distance from Neck Bearing (A) to Brake Wheel [m]

# Component Masses and Weights [kg]
m_rotor = 3500.0                        # Mass of rotor + sails + canister [kg]
m_shaft = 1500.0                        # Mass of main shaft [kg]
m_wheel = 1000.0                        # Mass of brake wheel [kg]
g = 9.81                                # Acceleration due to gravity [m/s²]

W_rotor = m_rotor * g                   # Rotor weight [N] (~34,335 N)
W_shaft = m_shaft * g                   # Shaft weight [N] (~14,715 N)
W_wheel = m_wheel * g                   # Brake wheel weight [N] (~9,810 N)

# Dynamic and Aerodynamic Loads
F_thrust = 9101.67                      # Aerodynamic wind thrust in extreme conditions [N]
Torque = 2463.51                        # Transmitted torque [N·m]

# Material Properties & Safety Factor
S_y = 250e6                             # Yield strength (e.g., cast iron / steel) [Pa]
n_safety = 2.5                          # Safety factor for static/fatigue design
S_allow = S_y / n_safety                # Allowable stress [Pa] (100 MPa)

# ==============================================================================
# 2. FORCE DECOMPOSITION (z-axis = along shaft, y-axis = radial)
# ==============================================================================

# Radial Components (perpendicular to shaft axis)
W_rotor_y = W_rotor * np.cos(alpha)
W_shaft_y = W_shaft * np.cos(alpha)
W_wheel_y = W_wheel * np.cos(alpha)
F_thrust_y = F_thrust * np.sin(alpha)   # Radial thrust component due to shaft angle

# Axial Components (parallel to shaft axis)
W_rotor_z = W_rotor * np.sin(alpha)
W_shaft_z = W_shaft * np.sin(alpha)
W_wheel_z = W_wheel * np.sin(alpha)
F_thrust_z = F_thrust * np.cos(alpha)

W_total_z = W_rotor_z + W_shaft_z + W_wheel_z
w_shaft_y = W_shaft_y / L                 # Distributed load from shaft self-weight [N/m]

# ==============================================================================
# 3. ANALYTICAL BEARING REACTION CALCULATIONS
# ==============================================================================

# A) Axial Reaction at Support B (Tail Thrust Bearing)
R_Bz = F_thrust_z + W_total_z

# B) Radial Reaction at Support B (Moment equilibrium about Support A = 0)
M_A_rotor = - W_rotor_y * l1
M_A_thrust = F_thrust_y * l1
M_A_shaft = w_shaft_y * L * (L / 2.0)
M_A_wheel = W_wheel_y * d1

R_By = (M_A_rotor + M_A_thrust + M_A_shaft + M_A_wheel) / L

# C) Radial Reaction at Support A (Radial force equilibrium in y-direction = 0)
R_Ay = (W_rotor_y - F_thrust_y + W_shaft_y + W_wheel_y) - R_By

print("=== CALCULATED BEARING REACTIONS ===")
print(f"Neck Bearing Radial Reaction (Support A): R_Ay = {R_Ay/1000:.3f} kN")
print(f"Tail Bearing Radial Reaction (Support B): R_By = {R_By/1000:.3f} kN")
print(f"Tail Bearing Axial Reaction  (Support B): R_Bz = {R_Bz/1000:.3f} kN")

# ==============================================================================
# 4. SHEAR FORCE V(z) & BENDING MOMENT M(z) DIAGRAMS
# ==============================================================================

z_vals = np.linspace(-l1, L, 1000)
V_vals = np.zeros_like(z_vals)
M_vals = np.zeros_like(z_vals)

for i, z in enumerate(z_vals):
    if z < 0:
        # Overhang section [-l1, 0)
        V = - W_rotor_y + F_thrust_y
        M = (- W_rotor_y + F_thrust_y) * (z + l1)
    elif z >= 0 and z < d1:
        # Section between Neck Bearing (A) and Brake Wheel [0, d1)
        V = (- W_rotor_y + F_thrust_y) + R_Ay - w_shaft_y * z
        M = (- W_rotor_y + F_thrust_y) * (z + l1) + R_Ay * z - w_shaft_y * z**2 / 2.0
    else:
        # Section between Brake Wheel and Tail Bearing (B) [d1, L]
        V = (- W_rotor_y + F_thrust_y) + R_Ay - w_shaft_y * z - W_wheel_y
        M = (- W_rotor_y + F_thrust_y) * (z + l1) + R_Ay * z - w_shaft_y * z**2 / 2.0 - W_wheel_y * (z - d1)
    
    V_vals[i] = V
    M_vals[i] = M

idx_max_M = np.argmax(np.abs(M_vals))
M_max = np.abs(M_vals[idx_max_M])
z_max_M = z_vals[idx_max_M]

print(f"\nMaximum Bending Moment: |M_max| = {M_max/1000:.3f} kN·m at z = {z_max_M:.2f} m")

# ==============================================================================
# 5. MINIMUM SHAFT DIAMETER CALCULATION (VON MISES / ASME CRITERION)
# ==============================================================================

# Equivalent bending moment combining Bending and Torsion
M_eq = np.sqrt(M_max**2 + 0.75 * Torque**2)

# Minimum diameter for solid circular cross-section: d = ((32 * n * M_eq) / (pi * S_y))^(1/3)
d_min = ((32 * n_safety * M_eq) / (np.pi * S_y))**(1/3)
d_min_mm = d_min * 1000

print(f"\n=== SHAFT DIAMETER SIZING ===")
print(f"Equivalent Moment (M_eq): {M_eq/1000:.3f} kN·m")
print(f"Required Minimum Diameter (d_min): {d_min_mm:.1f} mm ({d_min:.3f} m)")

# ==============================================================================
# 6. PLOTTING DIAGRAMS
# ==============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Main Windshaft Mechanical Analysis', fontsize=14, fontweight='bold')

# Shear Force Diagram
ax1.plot(z_vals, V_vals / 1000, color='#1f77b4', linewidth=2, label='Shear Force V(z)')
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.axvline(0, color='red', linestyle=':', label='Support A (Neck Bearing)')
ax1.axvline(d1, color='orange', linestyle=':', label='Brake Wheel')
ax1.axvline(L, color='purple', linestyle=':', label='Support B (Tail Bearing)')
ax1.fill_between(z_vals, V_vals / 1000, color='#1f77b4', alpha=0.15)
ax1.set_ylabel('Shear Force V [kN]', fontsize=11, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')
ax1.set_title('Shear Force Diagram (SFD)', fontsize=12)

# Bending Moment Diagram
ax2.plot(z_vals, M_vals / 1000, color='#d62728', linewidth=2, label='Bending Moment M(z)')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.axvline(0, color='red', linestyle=':')
ax2.axvline(d1, color='orange', linestyle=':')
ax2.axvline(L, color='purple', linestyle=':')
ax2.fill_between(z_vals, M_vals / 1000, color='#d62728', alpha=0.15)
ax2.plot(z_max_M, M_vals[idx_max_M] / 1000, 'ko', markersize=8, label=f'M_max = {M_vals[idx_max_M]/1000:.2f} kN·m')
ax2.set_xlabel('Position along shaft z-axis [m]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Bending Moment M [kN·m]', fontsize=11, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right')
ax2.set_title('Bending Moment Diagram (BMD)', fontsize=12)

plt.tight_layout()
plt.show()
