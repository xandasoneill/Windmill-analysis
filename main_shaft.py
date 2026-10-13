import numpy as np

from wind import Wind

# References (files in resources/):
#   [1] Condaxakis et al., "The Evolution of Windmill Design: From Lasithi Plateau Pumping Windmills
#       to Electricity Production", Energies 2026, 19, 829  (resources/The Evolution of Windmill Design_...pdf)
#   [2] Martin Bates, "Argos Hill Windmill" guide  (resources/Argos Hill Windmill Trust.pdf)


class MainShaft:
    """
    Static 3D analysis of the main windshaft (bearing reactions, shear force, bending moment and
    minimum diameter). Everything is computed on construction and stored as attributes.

    Frame: z along the shaft (rotor at z = -l1, neck bearing A at z = 0, tail bearing B at z = L),
    y radial in the vertical plane of the tilted shaft, x lateral.

    Parameters (defaults are the Argos Hill / extreme-wind case)
    ----------
    alpha_deg    : shaft inclination angle [deg]                      ref: [2] p.22 (neck bearing raised to 8°)
    beta_deg     : wind yaw misalignment angle [deg]
    gamma_deg    : wind vertical angle [deg] (>0 = updraft)
    l1           : overhang length (from neck bearing to rotor) [m]
    L            : distance between Neck (A) and Tail (B) [m]
    d1           : distance from Neck (A) to Brake Wheel [m]
    r_wheel      : pitch radius of Brake Wheel [m]                    (note: [2] p.11 gives brake wheel ~8 ft diameter -> ~1.2 m radius)
    phi_deg      : gear pressure angle [deg]                          (value comes from norms ISO, AGMA and DIN)
    m_rotor, m_shaft, m_wheel : masses [kg]
    g            : gravity [m/s²]
    F_wind_total : aerodynamic wind force [N]                         ref: [1] Sec. 3.3, p.22 (extreme case, 40 m/s, 80 RPM)
    Torque       : transmitted torque [N·m]                           ref: [1] Sec. 3.3, p.22 (Q = 2463.51 N·m, same extreme case)
    S_y          : yield strength [Pa]
    n_safety     : safety factor
    n_points     : number of points along z used for the diagrams

    Main results
    ------------
    F_wind_x/y/z, R_Ax/y, R_Bx/y/z, R_A_radial, R_B_radial
    z, Vx, Vy, V_res, Mx, My, M_res           (arrays along the shaft)
    M_max, z_max_M, M_eq, d_min
    supports : dict {label: z position} of the points to mark in the diagrams
    """

    def __init__(self,
                 alpha_deg=8.0, beta_deg=10.0, gamma_deg=0.0,
                 l1=0.8, L=4.88, d1=1.2, r_wheel=1.0, phi_deg=20.0,
                 m_rotor=3500.0, m_shaft=1500.0, m_wheel=1000.0, g=9.81,
                 F_wind_total=9101.67, Torque=2463.51,
                 S_y=250e6, n_safety=2.5, n_points=1000):
        
        self.alpha_deg, self.beta_deg, self.gamma_deg = alpha_deg, beta_deg, gamma_deg
        self.alpha = np.radians(alpha_deg)
        self.phi = np.radians(phi_deg)
        self.l1, self.L, self.d1, self.r_wheel = l1, L, d1, r_wheel
        self.m_rotor, self.m_shaft, self.m_wheel, self.g = m_rotor, m_shaft, m_wheel, g
        self.W_rotor, self.W_shaft, self.W_wheel = m_rotor * g, m_shaft * g, m_wheel * g
        self.F_wind_total, self.Torque = F_wind_total, Torque
        self.S_y, self.n_safety = S_y, n_safety
        self.S_allow = S_y / n_safety           # Allowable stress [Pa]
        self.n_points = n_points

        self.supports = {
            "Support A (Neck Bearing)": 0.0,
            "Brake Wheel": d1,
            "Support B (Tail Bearing)": L,
        }

        self._compute_loads()
        self._compute_reactions()
        self._compute_internal_loads()
        self._compute_diameter()

    # --------------------------------------------------------------------------
    # 1. Loads in the shaft frame (x lateral, y radial, z axial)
    # --------------------------------------------------------------------------
    def _compute_loads(self):
        alpha = self.alpha

        # Wind force components
        self.F_wind_x, self.F_wind_y, self.F_wind_z = Wind(
            self.F_wind_total, self.alpha_deg, self.beta_deg, self.gamma_deg).components

        # Gear reaction forces at z = d1
        self.F_tangential = self.Torque / self.r_wheel
        self.F_gear_x = -self.F_tangential
        self.F_gear_y = self.F_tangential * np.tan(self.phi)

        # Point load at the rotor (z = -l1): wind + weight
        self.F_rotor_x = self.F_wind_x
        self.F_rotor_y = self.F_wind_y - self.W_rotor * np.cos(alpha)
        self.F_rotor_z = self.F_wind_z - self.W_rotor * np.sin(alpha)

        # Point load at the brake wheel (z = d1): gear reaction + weight
        self.F_wheel_x = self.F_gear_x
        self.F_wheel_y = self.F_gear_y - self.W_wheel * np.cos(alpha)

        # Distributed load of the shaft self-weight [N/m]
        self.w_shaft_y = -(self.W_shaft / self.L) * np.cos(alpha)

    # --------------------------------------------------------------------------
    # 2. Bearing reactions
    # --------------------------------------------------------------------------
    def _compute_reactions(self):
        alpha, l1, L, d1 = self.alpha, self.l1, self.L, self.d1

        # Tail bearing (B): axial thrust + radial (moments about A = 0)
        self.R_Bz = self.F_wind_z + (self.W_rotor + self.W_shaft + self.W_wheel) * np.sin(alpha)
        self.R_Bx = (self.F_rotor_x * l1 - self.F_wheel_x * d1) / L
        self.R_By = (self.F_rotor_y * l1 - self.w_shaft_y * L**2 / 2.0 - self.F_wheel_y * d1) / L

        # Neck bearing (A): radial (force equilibrium)
        self.R_Ax = -(self.F_rotor_x + self.F_wheel_x + self.R_Bx)
        self.R_Ay = -(self.F_rotor_y + self.w_shaft_y * L + self.F_wheel_y + self.R_By)

        self.R_A_radial = np.sqrt(self.R_Ax**2 + self.R_Ay**2)
        self.R_B_radial = np.sqrt(self.R_Bx**2 + self.R_By**2)

    # --------------------------------------------------------------------------
    # 3. Shear force & bending moment along the shaft
    # --------------------------------------------------------------------------
    def _compute_internal_loads(self):
        l1, L, d1 = self.l1, self.L, self.d1
        F_rotor_x, F_rotor_y = self.F_rotor_x, self.F_rotor_y
        F_wheel_x, F_wheel_y = self.F_wheel_x, self.F_wheel_y
        R_Ax, R_Ay, w_shaft_y = self.R_Ax, self.R_Ay, self.w_shaft_y

        z_vals = np.linspace(-l1, L, self.n_points)
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

        self.z = z_vals
        self.Vx, self.Vy = Vx_vals, Vy_vals
        self.Mx, self.My = Mx_vals, My_vals
        self.V_res = np.sqrt(Vx_vals**2 + Vy_vals**2)
        self.M_res = np.sqrt(Mx_vals**2 + My_vals**2)

        idx_max_M = np.argmax(self.M_res)
        self.M_max = self.M_res[idx_max_M]
        self.z_max_M = z_vals[idx_max_M]

    # --------------------------------------------------------------------------
    # 4. Minimum shaft diameter (Von Mises / ASME criterion)
    # --------------------------------------------------------------------------
    def _compute_diameter(self):
        # Equivalent moment combining bending and torsion
        self.M_eq = np.sqrt(self.M_max**2 + 0.75 * self.Torque**2)

        # Solid circular cross-section: d = ((32 * n * M_eq) / (pi * S_y))^(1/3)
        self.d_min = ((32 * self.n_safety * self.M_eq) / (np.pi * self.S_y))**(1/3)

    # --------------------------------------------------------------------------
    # Output
    # --------------------------------------------------------------------------
    def report(self):
        """Print the bearing reactions, maximum moment and diameter sizing."""
        print("=== BEARING REACTIONS ===")
        print(f"Wind force (x, y, z): ({self.F_wind_x/1000:.3f}, {self.F_wind_y/1000:.3f}, {self.F_wind_z/1000:.3f}) kN")
        print(f"Neck Bearing A (radial): R_Ax = {self.R_Ax/1000:.3f} kN, R_Ay = {self.R_Ay/1000:.3f} kN -> {self.R_A_radial/1000:.3f} kN")
        print(f"Tail Bearing B (radial): R_Bx = {self.R_Bx/1000:.3f} kN, R_By = {self.R_By/1000:.3f} kN -> {self.R_B_radial/1000:.3f} kN")
        print(f"Tail Bearing B (axial):  R_Bz = {self.R_Bz/1000:.3f} kN")

        print(f"\nMaximum Resultant Bending Moment: M_max = {self.M_max/1000:.3f} kN·m at z = {self.z_max_M:.2f} m")

        print("\n=== SHAFT DIAMETER SIZING ===")
        print(f"Equivalent Moment (M_eq): {self.M_eq/1000:.3f} kN·m")
        print(f"Required Minimum Diameter (d_min): {self.d_min*1000:.1f} mm ({self.d_min:.3f} m)")
        # Sanity check: [2] p.19 gives the Argos Hill neck bearing as ~12 in (~305 mm) and tail bearing ~6 in (~152 mm)


if __name__ == "__main__":
    from visuals import plot_shear_moment

    shaft = MainShaft()
    shaft.report()
    plot_shear_moment(
        shaft.z,
        shear={"V_x": shaft.Vx, "V_y": shaft.Vy}, shear_res=shaft.V_res,
        moment={"M_x": shaft.Mx, "M_y": shaft.My}, moment_res=shaft.M_res,
        markers=shaft.supports,
        max_point=(shaft.z_max_M, shaft.M_max),
        title="Main Windshaft Mechanical Analysis",
    )
