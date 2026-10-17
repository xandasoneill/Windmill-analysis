import numpy as np


class Wind:
    """
    Wind force on the rotor, resolved in the local (tilted) shaft frame.

    Frame convention (same as main_shaft.py):
        z -> along the shaft, pointing from the rotor towards the tail bearing. The rotor end is
             raised by alpha, so +z points downhill: z = cos(alpha)*horizontal - sin(alpha)*up
        y -> perpendicular to the shaft, in the vertical plane: y = sin(alpha)*horizontal + cos(alpha)*up
        x -> lateral, perpendicular to the y-z plane

    Parameters
    ----------
    F_total   : wind force magnitude [N]
    alpha_deg : shaft tilt angle [deg]
    beta_deg  : yaw misalignment [deg] (0 = wind aligned with the shaft projection)
    gamma_deg : vertical wind angle [deg] (0 = horizontal, > 0 = updraft, < 0 = downdraft)

    Attributes
    ----------
    Fx, Fy, Fz : wind force components in the shaft frame [N]
    """

    def __init__(self, F_total, alpha_deg, beta_deg=0.0, gamma_deg=0.0):
        self.F_total = F_total
        self.alpha_deg = alpha_deg
        self.beta_deg = beta_deg
        self.gamma_deg = gamma_deg

        alpha = np.radians(alpha_deg)
        beta = np.radians(beta_deg)
        gamma = np.radians(gamma_deg)

        # Horizontal wind contribution (cos(gamma)) and vertical one (sin(gamma))
        horizontal = F_total * np.cos(gamma)
        vertical = F_total * np.sin(gamma)

        self.Fx = horizontal * np.sin(beta)
        self.Fy = vertical * np.cos(alpha) + horizontal * np.cos(beta) * np.sin(alpha)
        self.Fz = horizontal * np.cos(beta) * np.cos(alpha) - vertical * np.sin(alpha)

    @property
    def components(self):
        """(Fx, Fy, Fz) tuple, so it can be unpacked: Fx, Fy, Fz = wind.components"""
        return self.Fx, self.Fy, self.Fz

    def __repr__(self):
        return (f"Wind(F_total={self.F_total}, alpha_deg={self.alpha_deg}, "
                f"beta_deg={self.beta_deg}, gamma_deg={self.gamma_deg}) -> "
                f"Fx={self.Fx:.2f}, Fy={self.Fy:.2f}, Fz={self.Fz:.2f} N")
