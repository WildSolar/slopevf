"""Radiance scenes of an infinite multi-row array on sloped ground (ray-tracing reference).

Geometry is built directly as Radiance primitives (bifacial_radiance's scene builder
assumes horizontal ground). Lengths are in units of the panel height H (= 1 m).

Local frame: X along the rows, Y horizontal uphill, Z up. The world frame used by
Radiance is X = east, Y = north, Z = up; the local frame is rotated such that local
+Y points uphill, i.e. towards azimuth gamma + 180 deg (gamma = slope azimuth from
south, positive towards west, as in slopevf).

Scene variants ("level"):
  ideal  - matches the slopevf idealisations: thin black panels without gaps, Lambertian
           ground, and everything seen over the top edge of the downhill row is replaced
           by a uniform "far ground" of radiance GHI * albedo_fg / pi (a glow plane that is
           visible from above only and transparent to all rays from below).
  field  - as ideal, but without the far-ground plane: the view over the downhill row
           shows the actual field (rows and ground) and, below the slope, the ground glow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slopevf import Array
from slopevf.geometry import Geometry


# --- Radiance invocation --------------------------------------------------------------

def radiance_env():
    """Environment with Radiance on PATH; uses $RADIANCE_DIR if set, else the PATH."""
    env = os.environ.copy()
    rdir = env.get("RADIANCE_DIR")
    if rdir:
        env["PATH"] = str(Path(rdir) / "bin") + os.pathsep + env["PATH"]
        env["RAYPATH"] = os.pathsep.join([".", str(Path(rdir) / "lib")])
    if shutil.which("rtrace", path=env["PATH"]) is None:
        raise RuntimeError("Radiance not found: set RADIANCE_DIR or add Radiance/bin to PATH")
    return env


def run(cmd, cwd, env, stdin=None, stdout_file=None):
    exe = shutil.which(cmd[0], path=env["PATH"])   # Windows does not search the child's PATH
    cmd = [exe or cmd[0]] + list(cmd[1:])
    out = open(stdout_file, "wb") if stdout_file else subprocess.PIPE
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin, stdout=out, stderr=subprocess.PIPE)
    finally:
        if stdout_file:
            out.close()
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed: {p.stderr.decode(errors='replace')}")
    return p.stdout if not stdout_file else None


# --- geometry -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSize:
    rows_extent: float = 60.0      # rows are placed within +-rows_extent along the slope [H]
    row_half_length: float = 100.0
    ground_margin: float = 100.0   # ground extends this much beyond the outermost rows [H]


def _to_world(p, gamma_deg):
    g = np.radians(gamma_deg)
    X, Y, Z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([X * np.cos(g) + Y * np.sin(g), -X * np.sin(g) + Y * np.cos(g), Z], axis=-1)


def _polygon(mod, name, pts):
    coords = "\n".join(" ".join(f"{v:.6f}" for v in p) for p in pts)
    return f"{mod} polygon {name}\n0\n0\n{3 * len(pts)}\n{coords}\n"


def _box(mod, name, origin, a, b, c, gamma):
    """Parallelepiped spanned by vectors a, b, c (local frame) with outward-facing polygons."""
    o = np.asarray(origin, float)
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    if np.dot(np.cross(a, b), c) < 0:      # make (a, b, c) right-handed
        a, b = b, a
    faces = [(o, o + b, o + a + b, o + a), (o + c, o + a + c, o + a + b + c, o + b + c),
             (o, o + a, o + a + c, o + c), (o + b, o + b + c, o + a + b + c, o + a + b),
             (o, o + c, o + b + c, o + b), (o + a, o + a + b, o + a + b + c, o + a + c)]
    return "".join(_polygon(mod, f"{name}_{k}", _to_world(np.array(f), gamma)) for k, f in enumerate(faces))


def scene_rad(array: Array, level: str, albedo: float, ghi: float, albedo_fg: float,
              size: FieldSize = FieldSize(), thickness: float = 1e-5, panel_reflectance: float = 0.0,
              panel_diffuse: float = 0.0) -> str:
    """Radiance description of panels, ground and (level 'ideal') the far-ground plane.

    ``panel_reflectance``: perfectly specular panels; ``panel_diffuse``: Lambertian panels."""
    g = Geometry(array)
    bp, bg, rho, h, p = g.bp, g.bg, g.rho, g.h, g.p
    gam = array.gamma
    L = size.row_half_length
    n_rows = int(np.ceil(size.rows_extent / p))
    t_vec = np.array([0.0, np.cos(bp), np.sin(bp)])
    n_f = np.array([0.0, -np.sin(bp), np.cos(bp)])

    if panel_reflectance > 0:
        # perfectly specular, constant reflectance; Radiance creates exact virtual sources
        # (mirror images of the sun) for planar mirrors
        r = panel_reflectance
        panel = f"void mirror panel_mat\n0\n0\n3 {r} {r} {r}\n"
    elif panel_diffuse > 0:
        r = panel_diffuse
        panel = f"void plastic panel_mat\n0\n0\n5 {r} {r} {r} 0 0\n"
    else:
        panel = "void plastic panel_mat\n0\n0\n5 0 0 0 0 0\n"
    rad = [panel,
           f"void plastic ground_mat\n0\n0\n5 {albedo} {albedo} {albedo} 0 0\n"]
    for i in range(-n_rows, n_rows + 1):
        centre = np.array([0.0, i * rho, h + i * rho * np.tan(bg)])
        origin = centre - 0.5 * t_vec - 0.5 * thickness * n_f + np.array([-L, 0, 0])
        rad.append(_box("panel_mat", f"row{i}", origin, [2 * L, 0, 0], t_vec, thickness * n_f, gam))

    # ground plane Z = Y tan(bg) (normal pointing up: counter-clockwise seen from above)
    y_lo = -n_rows * rho - size.ground_margin
    y_hi = n_rows * rho + size.ground_margin
    xg = L + size.ground_margin
    pts = np.array([[-xg, y_lo, y_lo * np.tan(bg)], [xg, y_lo, y_lo * np.tan(bg)],
                    [xg, y_hi, y_hi * np.tan(bg)], [-xg, y_hi, y_hi * np.tan(bg)]])
    rad.append(_polygon("ground_mat", "ground", _to_world(pts, gam)))

    if level == "ideal":
        # far-ground plane through the upper edge of row -1, parallel to the slope, downhill
        top = np.array([0.0, -rho, h - rho * np.tan(bg)]) + 0.5 * t_vec
        lfg = ghi * albedo_fg / np.pi
        rad.append(f"void glow fg_glow\n0\n0\n4 {lfg:.6f} {lfg:.6f} {lfg:.6f} 0\n")
        rad.append("void mixfunc fg_mat\n4 fg_glow void from_above from_above.cal\n0\n0\n")
        y0, y1 = top[1] - 1e-4, y_lo - 1.0
        z = lambda y: top[2] + (y - top[1]) * np.tan(bg)  # noqa: E731
        pts = np.array([[-xg, y1, z(y1)], [xg, y1, z(y1)], [xg, y0, z(y0)], [-xg, y0, z(y0)]])
        rad.append(_polygon("fg_mat", "far_ground_plane", _to_world(pts, gam)))
    elif level != "field":
        raise ValueError(level)
    return "".join(rad)


FROM_ABOVE_CAL = "from_above = if(-Dz, 1, 0);\n"


def sensors(array: Array, chi, x_positions, offset=0.001, thickness=1e-5):
    """Sensor lines (position + direction, world frame) for front and back of row 0.

    Returns (front, back) arrays of shape (len(x_positions), len(chi), 6).
    """
    g = Geometry(array)
    bp, h = g.bp, g.h
    t_vec = np.array([0.0, np.cos(bp), np.sin(bp)])
    n_f = np.array([0.0, -np.sin(bp), np.cos(bp)])
    out = []
    for sign in (1.0, -1.0):
        n = sign * n_f
        P = (np.array([0.0, 0.0, h])[None, None, :]
             + np.asarray(x_positions)[:, None, None] * np.array([1.0, 0, 0])
             + np.asarray(chi)[None, :, None] * t_vec
             + (thickness / 2 + offset) * n)
        pw = _to_world(P, array.gamma)
        dw = np.broadcast_to(_to_world(n, array.gamma), pw.shape)
        out.append(np.concatenate([pw, dw], axis=-1))
    return out[0], out[1]


# --- sky ----------------------------------------------------------------------------------

PEREZ_CAL = """{ Perez (1993) all-weather sky radiance, no horizon blending (unlike perezlum.cal).
  A1: normalisation, A2..A6: Perez a..e, A7..A9: sun direction }
skybright = if(Dz, pos(A1 * (1 + A2*Exp(A3/Dz)) * (1 + A4*Exp(A5*gam) + A6*cg*cg)), 0);
pos(x) = if(x, x, 0);
cg = Dx*A7 + Dy*A8 + Dz*A9;
gam = Acos(if(cg - 1, 1, if(-1 - cg, -1, cg)));
"""
SUN_SOLID_ANGLE = 2 * np.pi * (1 - np.cos(np.radians(0.533 / 2)))


def perez_sky(sun_elevation, sun_azimuth, dni, dhi):
    """Perez a..e (as used by slopevf) and the normalisation for a continuous sky.

    The normalisation is obtained by fine numerical integration of the continuous
    radiance distribution such that the diffuse horizontal irradiance equals DHI.
    """
    import perezsky
    from slopevf.geometry import direction_vector
    el, az = np.radians(sun_elevation), np.radians(sun_azimuth)
    par = perezsky.perez_parameters(dni, dhi, 90.0 - sun_elevation).as_array()[:, 0]
    a, b, c, d, e = par
    sun = direction_vector(el, az)
    n_el, n_az = 3000, 3600
    ele = (np.arange(n_el) + 0.5) * (np.pi / 2) / n_el
    azi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az - np.pi
    total = 0.0
    for k in range(0, n_el, 300):
        E, A = np.meshgrid(ele[k:k + 300], azi, indexing="ij")
        v = direction_vector(E, A)
        cg = np.clip(v @ sun, -1, 1)
        lv = np.maximum(0, (1 + a * np.exp(b / np.sin(E))) * (1 + c * np.exp(d * np.arccos(cg)) + e * cg ** 2))
        total += np.sum(lv * np.sin(E) * np.cos(E)) * (np.pi / 2 / n_el) * (2 * np.pi / n_az)
    return par, (dhi / total if total > 0 else 0.0), sun


def sky_rad(sun_elevation, sun_azimuth, dni, dhi, ghi, albedo_fg) -> str:
    """Radiance sky identical to the slopevf sky model (point-sun mode), plus a uniform ground.

    Azimuths from south, positive towards west. Radiance world frame: x = east, y = north.
    """
    par, norm, sun = perez_sky(sun_elevation, sun_azimuth, dni, dhi)
    sx, sy, sz = -sun[0], -sun[1], sun[2]
    lg = ghi * albedo_fg / np.pi
    txt = ("void brightfunc skyfunc\n2 skybright slopevf_sky.cal\n0\n"
           f"9 {norm:.8e} " + " ".join(f"{p:.8f}" for p in par) + f" {sx:.8f} {sy:.8f} {sz:.8f}\n"
           "skyfunc glow sky_glow\n0\n0\n4 1 1 1 0\n"
           "sky_glow source sky\n0\n0\n4 0 0 1 180\n"
           f"void glow ground_glow\n0\n0\n4 {lg:.6f} {lg:.6f} {lg:.6f} 0\n"
           "ground_glow source ground\n0\n0\n4 0 0 -1 180\n")
    if dni > 0:
        ls = dni / SUN_SOLID_ANGLE
        txt += (f"void light solar\n0\n0\n3 {ls:.6e} {ls:.6e} {ls:.6e}\n"
                f"solar source sun\n0\n0\n4 {sx:.8f} {sy:.8f} {sz:.8f} 0.533\n")
    return txt


# --- ray tracing ----------------------------------------------------------------------------

# -ab 2 is exact for the 'ideal'/'field' levels (black panels): sky -> ground -> sensor.
# The pilot showed no change with -ab 4; -ad 16384 keeps the per-sensor noise well below 0.5%.
DEFAULT_RTRACE = ["-ab", "2", "-aa", "0", "-ad", "16384", "-as", "0", "-lw", "1e-7", "-lr", "8"]


def irradiance(octree, rays, workdir, env, params=DEFAULT_RTRACE, n_proc=1):
    """Irradiance [W/m^2] for an (N, 6) array of sensor rays (rtrace -I), in parallel chunks."""
    rays = np.asarray(rays).reshape(-1, 6)
    chunks = np.array_split(np.arange(len(rays)), max(1, min(n_proc, len(rays))))

    def one(idx):
        txt = "\n".join(" ".join(f"{v:.6f}" for v in r) for r in rays[idx]) + "\n"
        out = run(["rtrace", "-h", "-I"] + list(params) + [str(octree)], workdir, env, stdin=txt.encode())
        vals = np.array(out.split(), dtype=float).reshape(-1, 3)
        return vals @ np.array([0.265, 0.670, 0.065])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(chunks)) as ex:
        parts = list(ex.map(one, chunks))
    return np.concatenate(parts)


def build_octree(workdir, env, name, rad_texts: dict) -> Path:
    workdir = Path(workdir)
    files = []
    for fname, txt in rad_texts.items():
        (workdir / fname).write_text(txt)
        files.append(fname)
    (workdir / "from_above.cal").write_text(FROM_ABOVE_CAL)
    (workdir / "slopevf_sky.cal").write_text(PEREZ_CAL)
    oct_path = workdir / f"{name}.oct"
    run(["oconv"] + files, workdir, env, stdout_file=oct_path)
    return oct_path
