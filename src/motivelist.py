# vim: set expandtab ts=4 sw=4:
"""Reading/writing TOM/AV3 ".em" motive lists and computing particle placements.

A motive list (MOTIVELIST) is a (20, N) matrix, one column per particle.  The
row meanings follow the TOM/AV3 toolbox convention used by the original Place
Object plugin (rows are 1-indexed in the documentation, 0-indexed here):

    row 0  : cross-correlation coefficient (CCC)
    rows 7-9   : X/Y/Z coordinate in the full tomogram
    rows 10-12 : X/Y/Z shift within the subvolume
    rows 16-18 : phi / psi / theta Euler angles (ZXZ, degrees)
    row 19 : class number

The absolute particle position is (coordinate + shift); we store it in
rows 13-15 for convenience, matching the original plugin.
"""

import numpy as np

# Row indices (0-based) into the (20, N) motive-list matrix.
ROW_CCC = 0
ROWS_COORD = slice(7, 10)     # X, Y, Z in full tomogram
ROWS_SHIFT = slice(10, 13)    # X, Y, Z shift in subvolume
ROWS_POSITION = slice(13, 16)  # computed absolute position (coord + shift)
ROWS_ANGLES = slice(16, 19)   # phi, psi, theta (ZXZ, degrees)
ROW_CLASS = 19
N_ROWS = 20


# ---------------------------------------------------------------------------
# EM file IO (subset of the PyTom .em format, ported from the original qio.py)
# ---------------------------------------------------------------------------
def read_em_motivelist(path):
    """Read a motive list from an ``.em`` file and return a (20, N) float array.

    Only little-endian float32/short/long/double data is supported, matching
    the original plugin.
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, np.dtype("int32"), 128)
        x, y, z = int(header[1]), int(header[2]), int(header[3])

        # The data type is encoded in the third hex digit of header[0].
        dt = int(hex(int(header[0]))[2])
        if dt == 2:        # short
            dt_data = np.dtype("<i2")
        elif dt == 4:      # long
            dt_data = np.dtype("<i4")
        elif dt == 5:      # float32
            dt_data = np.dtype("<f4")
        elif dt == 9:      # double
            dt_data = np.dtype("<f8")
        else:
            raise ValueError("Unsupported EM data type code %d" % dt)

        v = np.fromfile(f, dt_data, x * y * z)

    volume = np.asarray(v.reshape((x, y, z), order="F"), dtype=np.float64)

    # A motive list is stored as x=20 rows, y=N particles, z=1.
    if x != N_ROWS:
        raise ValueError(
            "Not a valid motive list: expected %d rows, got %d" % (N_ROWS, x))
    motl = volume[:, :, 0] if z == 1 else volume.reshape((x, -1), order="F")
    return np.ascontiguousarray(motl)


def write_em_motivelist(path, motl):
    """Write a (20, M) motive-list matrix to an ``.em`` float32 file."""
    data = np.asarray(motl, dtype=np.float32)
    header = np.zeros(128, dtype="int32")
    header[0] = 83886086  # 0x5000006: little-endian float32, version byte
    if data.ndim == 2:
        header[1], header[2] = data.shape
        header[3] = 1
    elif data.ndim == 3:
        header[1:4] = data.shape
    else:
        raise ValueError("motive list must be 2D or 3D")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asfortranarray(data).tobytes(order="F"))


def prepare_motivelist(motl):
    """Validate a raw (20, N) motive list and fill in the absolute positions.

    Returns a copy of the matrix with rows 13-15 set to coordinate + shift.
    Raises ``ValueError`` if the matrix does not have 20 rows.
    """
    motl = np.array(motl, dtype=np.float64, copy=True)
    if motl.ndim != 2 or motl.shape[0] != N_ROWS:
        raise ValueError(
            "Motive list must have %d rows; got shape %s" % (N_ROWS, motl.shape))
    motl[ROWS_POSITION, :] = motl[ROWS_COORD, :] + motl[ROWS_SHIFT, :]
    return motl


# ---------------------------------------------------------------------------
# Rotation maths (ZXZ Euler convention, ported from the original qtools.py)
# ---------------------------------------------------------------------------
def _stack_rot_z(angles_rad):
    """Vectorised rotation matrices about Z for an array of angles (radians)."""
    c, s = np.cos(angles_rad), np.sin(angles_rad)
    n = angles_rad.shape[0]
    m = np.zeros((n, 3, 3))
    m[:, 0, 0] = c
    m[:, 0, 1] = -s
    m[:, 1, 0] = s
    m[:, 1, 1] = c
    m[:, 2, 2] = 1.0
    return m


def _stack_rot_x(angles_rad):
    """Vectorised rotation matrices about X for an array of angles (radians)."""
    c, s = np.cos(angles_rad), np.sin(angles_rad)
    n = angles_rad.shape[0]
    m = np.zeros((n, 3, 3))
    m[:, 0, 0] = 1.0
    m[:, 1, 1] = c
    m[:, 1, 2] = -s
    m[:, 2, 1] = s
    m[:, 2, 2] = c
    return m


def rotation_matrices_zxz(angles_deg):
    """Return an (N, 3, 3) stack of ZXZ rotation matrices.

    ``angles_deg`` is a (3, N) array of [phi, psi, theta] columns, matching the
    motive-list layout.  The convention matches the original plugin:
    R = Rz(psi) @ Rx(theta) @ Rz(phi).
    """
    phi = np.deg2rad(angles_deg[0])
    psi = np.deg2rad(angles_deg[1])
    theta = np.deg2rad(angles_deg[2])
    zm1 = _stack_rot_z(phi)
    xm = _stack_rot_x(theta)
    zm2 = _stack_rot_z(psi)
    return np.matmul(zm2, np.matmul(xm, zm1))


def placement_array(motl, voxel_size=1.0, z_offset=0.0, phi_offset=0.0):
    """Build an (N, 3, 4) array of placement transforms for the particles.

    Each transform maps the base object geometry to global tomogram coordinates:
    the 3x3 linear block is ``voxel_size * R`` and the final column is the
    particle position (plus an optional offset along the rotated Z axis).
    Suitable for ``chimerax.geometry.Places(place_array=...)``.
    """
    n = motl.shape[1]
    angles = np.array(motl[ROWS_ANGLES, :], dtype=np.float64, copy=True)
    if phi_offset:
        angles[0] += phi_offset
    rot = rotation_matrices_zxz(angles)            # (N, 3, 3)
    positions = motl[ROWS_POSITION, :].T.copy()    # (N, 3)
    if z_offset:
        # rotated +Z axis is the third column of each rotation matrix
        positions = positions + z_offset * rot[:, :, 2]
    pa = np.zeros((n, 3, 4), dtype=np.float64)
    pa[:, :, :3] = rot * float(voxel_size)
    pa[:, :, 3] = positions
    return np.ascontiguousarray(pa)


# ---------------------------------------------------------------------------
# Single 3x3 rotation matrices (ported from the original qtools.py), used by
# the geometric picking code.
# ---------------------------------------------------------------------------
def rotation_matrix_x(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_matrix_y(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_matrix_z(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotation_matrix_zxz(angles_deg):
    """ZXZ matrix for a single [phi, psi, theta] triple = Rz(psi)Rx(theta)Rz(phi)."""
    z1, z2, x = angles_deg
    return rotation_matrix_z(z2) @ rotation_matrix_x(x) @ rotation_matrix_z(z1)


def normal_to_zxz(normal):
    """Return (psi, theta) degrees so the object +Z axis aligns with ``normal``.

    Matches the sphere convention of the original Pick Particle plugin:
    psi = deg(azimuth) - 270, theta = 90 - deg(elevation).  Phi (in-plane spin)
    is left free for the caller (randomised or a helical twist).
    """
    n = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm == 0:
        return 0.0, 0.0
    n = n / norm
    az = np.arctan2(n[1], n[0])
    el = np.arcsin(np.clip(n[2], -1.0, 1.0))
    psi = np.rad2deg(az) - 270.0
    theta = 90.0 - np.rad2deg(el)
    return psi, theta


def natural_cubic_spline(points, n_samples):
    """Sample a natural cubic spline through ``points`` (M,3).

    Returns (positions (n_samples,3), unit_tangents (n_samples,3)).
    Falls back to linear interpolation for 2 control points.
    """
    pts = np.asarray(points, dtype=np.float64)
    m = len(pts)
    if m < 2:
        t = np.tile(np.array([0.0, 0.0, 1.0]), (max(m, 1), 1))
        return pts.copy(), t
    u = np.linspace(0.0, m - 1, n_samples)
    if m == 2:
        s = u / (m - 1)
        pos = pts[0] * (1 - s)[:, None] + pts[1] * s[:, None]
        tan = np.tile(pts[1] - pts[0], (n_samples, 1))
    else:
        from scipy.interpolate import CubicSpline
        x = np.arange(m)
        cs = CubicSpline(x, pts, bc_type="natural")
        pos = cs(u)
        tan = cs(u, 1)
    norms = np.linalg.norm(tan, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pos, tan / norms


# ---------------------------------------------------------------------------
# Motive-list utilities (ported from the original Pick Particle Options panel)
# ---------------------------------------------------------------------------
def combine_motls(motls, renumber=False):
    """Concatenate several (20, N) motive lists into one.

    The feature-index row (row 6) is offset cumulatively between files, matching
    the original 'Combine Coordinates' behaviour.  If ``renumber`` is True the
    particle-number row (row 4) is renumbered sequentially.
    """
    combined = np.zeros((N_ROWS, 0), dtype=np.float64)
    offset = 0.0
    for motl in motls:
        m = np.array(motl, dtype=np.float64, copy=True)
        if m.shape[0] != N_ROWS or m.shape[1] == 0:
            continue
        m[5, :] += offset
        offset = m[5, -1]
        combined = np.concatenate((combined, m), axis=1)
    if renumber and combined.shape[1] > 0:
        combined[3, :] = np.arange(1, combined.shape[1] + 1)
    return combined


def shift_motls(motl, shift=(0.0, 0.0, 0.0), apix_in=1.0, apix_out=1.0):
    """Translate each particle by a rotated offset and optionally rescale/bin.

    Port of the original 'Shift Coordinates' tool.
    """
    m = np.array(motl, dtype=np.float64, copy=True)
    sx, sy, sz = shift
    if sx or sy or sz:
        s = np.array([sx, sy, sz], dtype=np.float64)
        rot = rotation_matrices_zxz(m[ROWS_ANGLES, :])      # (N,3,3)
        rshift = np.einsum("nij,j->ni", rot, s)             # (N,3)
        m[ROWS_SHIFT, :] += rshift.T
    m[ROWS_COORD, :] += m[ROWS_SHIFT, :]
    if apix_in != apix_out:
        binfac = apix_in / apix_out
        m[ROWS_COORD, :] = m[ROWS_COORD, :] * binfac - (binfac - 1)
    m[ROWS_SHIFT, :] = m[ROWS_COORD, :] - np.round(m[ROWS_COORD, :])
    m[ROWS_COORD, :] = np.round(m[ROWS_COORD, :])
    return m


# ---------------------------------------------------------------------------
# STOPGAP star export (write-only)
# ---------------------------------------------------------------------------
_STOPGAP_COLUMNS = (
    "_motl_idx", "_tomo_num", "_object", "_subtomo_num", "_halfset",
    "_orig_x", "_orig_y", "_orig_z", "_score",
    "_x_shift", "_y_shift", "_z_shift", "_phi", "_psi", "_the", "_class",
)


def write_stopgap_star(path, motl):
    """Write a (20, N) TOM/AV3 motive list as a STOPGAP motivelist .star file."""
    m = np.asarray(motl, dtype=np.float64)
    n = m.shape[1]
    idx = np.arange(1, n + 1)
    obj = m[5, :].copy()
    obj[obj == 0] = 1
    halfset = (idx % 2 == 0).astype(int) + 1     # alternate 1/2
    cols = {
        "_motl_idx": idx,
        "_tomo_num": m[4, :].astype(int),
        "_object": obj.astype(int),
        "_subtomo_num": m[3, :].astype(int),
        "_halfset": halfset,
        "_orig_x": m[7, :], "_orig_y": m[8, :], "_orig_z": m[9, :],
        "_score": m[ROW_CCC, :],
        "_x_shift": m[10, :], "_y_shift": m[11, :], "_z_shift": m[12, :],
        "_phi": m[16, :], "_psi": m[17, :], "_the": m[18, :],
        "_class": m[ROW_CLASS, :].astype(int),
    }
    int_cols = {"_motl_idx", "_tomo_num", "_object", "_subtomo_num",
                "_halfset", "_class"}
    with open(path, "w") as f:
        f.write("data_stopgap_motivelist\n\nloop_\n")
        for i, name in enumerate(_STOPGAP_COLUMNS, start=1):
            f.write("%s #%d\n" % (name, i))
        for j in range(n):
            row = ["%d" % int(cols[name][j]) if name in int_cols
                   else "%.6f" % float(cols[name][j])
                   for name in _STOPGAP_COLUMNS]
            f.write("\t".join(row) + "\n")
