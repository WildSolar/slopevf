"""Python counterpart of example_script.m of the MATLAB implementation
(https://github.com/WildSolar/General_PV_Irradiation_Model)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slopevf import Array, FarGround, Settings, simulate

# --- geometric configuration (paper notation) -----------------------------------------
array = Array(
    beta_p=60,        # panel tilt [deg]
    beta_g=20,        # ground slope [deg]
    rho_bar=1.125,    # horizontal row spacing / panel height [-]
    h_bar=1.84,       # panel centre height above ground / panel height [-]
    gamma=0,          # ground azimuth [deg] (0 = south, 90 = west)
)
far_ground = FarGround(beta=20, azimuth=160)   # characteristic far-ground slope and azimuth [deg]
print(f"shading angle: {array.shading_angle:.1f} deg")

# --- irradiation: any number of situations (or a time-indexed DataFrame) ----------------
weather = dict(
    dni=np.linspace(550, 600, 5),                  # direct normal irradiance [W/m^2]
    ghi=np.linspace(200, 400, 5),                  # global horizontal irradiance [W/m^2]
    solar_azimuth=np.linspace(-40, -20, 5),        # [deg] (0 = south, 90 = west)
    solar_elevation=np.linspace(11, 16, 5),        # [deg]
)

# --- simulation ------------------------------------------------------------------------
# Settings() uses the corrected model; Settings.matlab() reproduces the MATLAB code.
res = simulate(array, weather, albedo=0.79, far_ground=far_ground, albedo_fg=0.3,
               settings=Settings(n_chi=100))

print(res.summary().round(1).T)

# res.front / res.back: (n_situations, n_chi) total irradiance along the panel [W/m^2]
# res.front_direct, .front_nearground, .front_farground, .back_direct, .back_nearground:
#   individual contributions; res.chi: panel positions (-0.5 = lower edge, 0.5 = upper edge)

out = Path("output")
out.mkdir(exist_ok=True)
for i in range(len(res.index)):
    ax = res.plot(i)
    ax.figure.tight_layout()
    ax.figure.savefig(out / f"example_{i + 1}.png", dpi=200)
    plt.close(ax.figure)
