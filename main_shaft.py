import numpy as np

from wind import Wind

# References (files in resources/):
#   [1] Condaxakis et al., "The Evolution of Windmill Design: From Lasithi Plateau Pumping Windmills
#       to Electricity Production", Energies 2026, 19, 829  (resources/The Evolution of Windmill Design_...pdf)
#   [2] Martin Bates, "Argos Hill Windmill" guide  (resources/Argos Hill Windmill Trust.pdf)


class MainShaft:
    """
    Static 3D analysis of the main windshaft (bearing reactions, shear force, bending moment and
    minimum diameter), following the free body diagram in main_shaft.svg. Everything is computed on
    construction and stored as attributes.

    Frame: z along the shaft, pointing from the rotor tip towards the tail bearing (rotor tip at
    z = -l1, neck bearing A at z = 0, brake wheel at z = d1, tail bearing B at z = L = d1 + d2).
    The rotor end is raised by alpha ([2] p.22), so +z points downhill. y is perpendicular to the
    shaft in its vertical plane (upwards), x is lateral.

    Loads (free body diagram):
        F_wind              at z = -l1    wind on the rotor tip (Wind class, angles beta/gamma)
        F_C = (Cx, Cy, Cz)  at z = -l1/2  force from the sails: rotor weight + F_C_aero
        W_shaft (overhang)  at z = -l1/2  shaft weight between tip and A, lumped at its centroid
        W_shaft (A -> B)    on [0, L]     distributed shaft weight
        W_BrakeWheel        at z = d1     brake wheel weight
        F_t (x), F_r (y)    at z = d1     gear forces from the pinion meshing with the brake wheel
        R_A = (Ax, Ay)      at z = 0      neck bearing (radial only)
        R_B = (Bx, By, Bz)  at z = L      tail bearing (radial + axial thrust)

    Parameters (defaults are the Argos Hill / extreme-wind case)
    ----------
    alpha_deg    : shaft inclination angle [deg]                      ref: [2] p.22 (neck bearing raised to 8°)
    beta_deg     : wind yaw misalignment angle [deg]
    gamma_deg    : wind vertical angle [deg] (>0 = updraft)
    l1           : distance from neck bearing A to the rotor tip [m]
    L            : distance between Neck (A) and Tail (B) [m]
    d1           : distance from Neck (A) to Brake Wheel [m]  (d2 = L - d1, Brake Wheel to Tail)
    r_wheel      : pitch radius of Brake Wheel [m]                    (note: [2] p.11 gives brake wheel ~8 ft diameter -> ~1.2 m radius)
    phi_deg      : gear pressure angle [deg]                          (value comes from norms ISO, AGMA and DIN)
    m_rotor, m_shaft, m_wheel : masses [kg]  (m_rotor = sails + canister, carried by F_C)
    g            : gravity [m/s²]
    F_wind_total : aerodynamic wind force [N]                         ref: [1] Sec. 3.3, p.22 (extreme case, 40 m/s, 80 RPM)
    F_C_aero     : (x, y, z) aerodynamic force from the sails added to F_C [N], in the shaft frame
    Torque       : transmitted torque [N·m]                           ref: [1] Sec. 3.3, p.22 (Q = 2463.51 N·m, same extreme case)
    S_y          : yield strength [Pa]
    n_safety     : safety factor
    n_points     : number of points along z used for the diagrams

    Main results
    ------------
    F_wind_x/y/z, F_Cx/y/z, F_t, F_r, R_Ax/y, R_Bx/y/z, R_A_radial, R_B_radial
    z, Vx, Vy, V_res, Mx, My, M_res           (arrays along the shaft)
    M_max, z_max_M, M_eq, d_min
    point_loads : list of (label, z, (Fx, Fy, Fz)) applied loads, reactions excluded
    supports    : dict {label: z position} of the bearings and brake wheel
    markers     : supports + load points, to mark in the diagrams
    Reactions are forces on the shaft; e.g. R_Bz < 0 means the tail bearing pushes towards the rotor.
    """

    def __init__(self,
                 alpha_deg=8.0, beta_deg=10.0, gamma_deg=0.0,
                 l1=0.8, L=4.88, d1=1.2, r_wheel=1.0, phi_deg=20.0,
                 m_rotor=3500.0, m_shaft=1500.0, m_wheel=1000.0, g=9.81,
                 F_wind_total=9101.67, F_C_aero=(0.0, 0.0, 0.0), Torque=2463.51,
                 S_y=250e6, n_safety=2.5, n_points=1000):

        self.alpha_deg, self.beta_deg, self.gamma_deg = alpha_deg, beta_deg, gamma_deg
        self.alpha = np.radians(alpha_deg)
        self.phi = np.radians(phi_deg)
        self.l1, self.L, self.d1, self.r_wheel = l1, L, d1, r_wheel
        self.d2 = L - d1
        self.m_rotor, self.m_shaft, self.m_wheel, self.g = m_rotor, m_shaft, m_wheel, g
        self.W_rotor, self.W_shaft, self.W_wheel = m_rotor * g, m_shaft * g, m_wheel * g
        self.F_wind_total, self.Torque = F_wind_total, Torque
        self.F_C_aero = np.asarray(F_C_aero, dtype=float)
        self.S_y, self.n_safety = S_y, n_safety
        self.S_allow = S_y / n_safety           # Allowable stress [Pa]
        self.n_points = n_points

        self.supports = {
            "Support A (Neck Bearing)": 0.0,
            "Brake Wheel": d1,
            "Support B (Tail Bearing)": L,
        }
        self.markers = {"Rotor tip (F_wind)": -l1, "Sails (F_C)": -l1 / 2.0, **self.supports}

        self._compute_loads()
        self._compute_reactions()
        self._compute_internal_loads()
        self._compute_diameter()

    # --------------------------------------------------------------------------
    # 1. Loads in the shaft frame (x lateral, y radial, z axial)
    # --------------------------------------------------------------------------
    def _compute_loads(self):
        l1, L, d1 = self.l1, self.L, self.d1

        # Direction of gravity in the shaft frame (rotor end raised -> weight pushes towards the tail)
        gravity = np.array([0.0, -np.cos(self.alpha), np.sin(self.alpha)])

        # Wind on the rotor tip (z = -l1)
        self.F_wind = np.array(Wind(self.F_wind_total, self.alpha_deg,
                                    self.beta_deg, self.gamma_deg).components)
        self.F_wind_x, self.F_wind_y, self.F_wind_z = self.F_wind

        # Force from the sails (z = -l1/2): rotor weight + aerodynamic part
        self.F_C = self.W_rotor * gravity + self.F_C_aero
        self.F_Cx, self.F_Cy, self.F_Cz = self.F_C

        # Shaft weight: overhang part lumped at its centroid, the rest distributed between A and B
        W_shaft_overhang = self.W_shaft * (l1 / (L + l1))
        W_shaft_between = self.W_shaft * (L / (L + l1))
        self.w_shaft = (W_shaft_between / L) * gravity      # distributed load [N/m], (x, y, z)

        # Gear forces at the brake wheel (z = d1): tangential (x) and radial (y)
        self.F_t = self.Torque / self.r_wheel
        self.F_r = self.F_t * np.tan(self.phi)
        F_gear = np.array([-self.F_t, self.F_r, 0.0])

        self.point_loads = [
            ("F_wind",         -l1,       self.F_wind),
            ("F_C (sails)",    -l1 / 2.0, self.F_C),
            ("W_shaft (overhang)", -l1 / 2.0, W_shaft_overhang * gravity),
            ("W_BrakeWheel",   d1,        self.W_wheel * gravity),
            ("F_t, F_r (gear)", d1,       F_gear),
        ]

    # --------------------------------------------------------------------------
    # 2. Bearing reactions
    # --------------------------------------------------------------------------
    def _compute_reactions(self):
        L = self.L
        w = self.w_shaft

        # Resultant force and moment arm sums of the applied loads (moments about A, z = 0)
        F_sum = sum(F for _, _, F in self.point_loads) + w * L
        Fz_arm_sum = sum(F * z for _, z, F in self.point_loads) + w * L**2 / 2.0

        # Tail bearing B: moment equilibrium about A in the y-z and x-z planes, all the axial load
        self.R_Bx = -Fz_arm_sum[0] / L
        self.R_By = -Fz_arm_sum[1] / L
        self.R_Bz = -F_sum[2]

        # Neck bearing A: radial force equilibrium
        self.R_Ax = -(F_sum[0] + self.R_Bx)
        self.R_Ay = -(F_sum[1] + self.R_By)

        self.R_A_radial = np.hypot(self.R_Ax, self.R_Ay)
        self.R_B_radial = np.hypot(self.R_Bx, self.R_By)

    # --------------------------------------------------------------------------
    # 3. Shear force & bending moment along the shaft
    # --------------------------------------------------------------------------
    def _compute_internal_loads(self):
        z = np.linspace(-self.l1, self.L, self.n_points)
        Vx, Vy = np.zeros_like(z), np.zeros_like(z)
        Mx, My = np.zeros_like(z), np.zeros_like(z)

        # Cutting the shaft at z and summing every force on the left of the cut
        loads = self.point_loads + [("R_A", 0.0, np.array([self.R_Ax, self.R_Ay, 0.0]))]
        for _, z_i, F in loads:
            left = z >= z_i
            arm = np.where(left, z - z_i, 0.0)
            Vx += np.where(left, F[0], 0.0)
            Vy += np.where(left, F[1], 0.0)
            Mx += F[1] * arm
            My -= F[0] * arm

        # Distributed shaft weight between A and B
        z_in = np.clip(z, 0.0, self.L)
        Vy += self.w_shaft[1] * z_in
        Mx += self.w_shaft[1] * z_in**2 / 2.0

        self.z = z
        self.Vx, self.Vy = Vx, Vy
        self.Mx, self.My = Mx, My
        self.V_res = np.hypot(Vx, Vy)
        self.M_res = np.hypot(Mx, My)

        # Equilibrium check: the bending moment must vanish at the tail bearing (free end of the beam)
        assert np.isclose(self.M_res[-1], 0.0, atol=1e-6 * max(1.0, self.M_res.max())), \
            "Moment equilibrium not satisfied at the tail bearing"

        idx_max_M = np.argmax(self.M_res)
        self.M_max = self.M_res[idx_max_M]
        self.z_max_M = z[idx_max_M]

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
        """Print the applied loads, bearing reactions, maximum moment and diameter sizing."""
        print("=== APPLIED LOADS (shaft frame, kN) ===")
        for label, z_i, F in self.point_loads:
            print(f"{label:20s} z = {z_i:+.2f} m   ({F[0]/1000:+.3f}, {F[1]/1000:+.3f}, {F[2]/1000:+.3f})")
        print(f"{'w_shaft (A -> B)':20s} [0, {self.L:.2f}] m  "
              f"({self.w_shaft[0]/1000:+.3f}, {self.w_shaft[1]/1000:+.3f}, {self.w_shaft[2]/1000:+.3f}) kN/m")

        print("\n=== BEARING REACTIONS ===")
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
        markers=shaft.markers,
        max_point=(shaft.z_max_M, shaft.M_max),
        title="Main Windshaft Mechanical Analysis",
    )
