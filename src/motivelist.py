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


def bake_offsets(motl, z_offset=0.0, phi_offset=0.0):
    """Return a copy of ``motl`` with a display Z/phi offset folded into it.

    ``z_offset`` moves each particle along its own +Z axis (voxels); ``phi_offset``
    is added to the in-plane phi angle.  Mirrors the Place Object display
    transform so an exported list can match what is shown.
    """
    m = np.array(motl, dtype=np.float64, copy=True)
    if phi_offset:
        m[16, :] = m[16, :] + phi_offset
    if z_offset:
        zdir = rotation_matrices_zxz(m[ROWS_ANGLES, :])[:, :, 2].T   # (3, N)
        abspos = m[ROWS_COORD, :] + m[ROWS_SHIFT, :] + z_offset * zdir
        m[ROWS_SHIFT, :] = abspos - np.round(abspos)
        m[ROWS_COORD, :] = np.round(abspos)
    return m


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
    the original 'Combine Coordinates' behavior.  If ``renumber`` is True the
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
    # Match STOPGAP's own stopgap_star_write.m output exactly, because
    # stopgap_star_read.m is stricter than the RELION dialect we used before:
    #  1. Column tags are written bare ("_motl_idx"), NOT with a RELION-style
    #     "#N" column-index suffix. In the STAR standard "#" begins a comment,
    #     and STOPGAP's reader does not expect the "#N" form.
    #  2. A blank line separates the loop_ header from the data rows. STOPGAP's
    #     reader selects data lines with `index > header_length + 1` (header_length
    #     = line of the last tag), so without this blank line the FIRST particle
    #     would be silently dropped.
    with open(path, "w") as f:
        f.write("\ndata_stopgap_motivelist\n\nloop_\n")
        for name in _STOPGAP_COLUMNS:
            f.write("%s\n" % name)
        f.write("\n")
        for j in range(n):
            row = ["%d" % int(cols[name][j]) if name in int_cols
                   else "%.6f" % float(cols[name][j])
                   for name in _STOPGAP_COLUMNS]
            f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Dynamo / RELION Euler-angle conversions
#
# Geopickr stores each orientation as ZXZ (phi, psi, theta) with the active
# matrix R = Rz(psi) Rx(theta) Rz(phi) (object +Z -> particle axis/normal).
# We convert by extracting Euler angles from R, which preserves the full
# orientation (including in-plane phi/twist) that Place Object displays.
#
# Ported from the validated reference scripts in dynamoForMatlab/
# (surface_to_dynamo_table.py, surface_to_relion_star.py).
# ---------------------------------------------------------------------------
def tom_rotation_matrices(motl):
    """(N,3,3) stack of rotation matrices for the particles in ``motl``."""
    return rotation_matrices_zxz(motl[ROWS_ANGLES, :])


def dynamo_matrix2euler(R):
    """Extract Dynamo ZXZ Euler angles (tdrot, tilt, narot) in degrees.

    Accepts a single (3,3) matrix or an (N,3,3) stack; returns (3,) or (N,3).
    Direct translation of dynamo_matrix2euler.m.
    """
    R = np.asarray(R, dtype=np.float64)
    single = (R.ndim == 2)
    if single:
        R = R[None, :, :]
    n = R.shape[0]
    r33 = np.clip(R[:, 2, 2], -1.0, 1.0)
    tdrot = np.zeros(n)
    tilt = np.zeros(n)
    narot = np.zeros(n)
    tol = 1e-9                            # only true gimbal poles use the branch
    m1 = np.abs(r33 - 1) < tol            # tilt ~ 0
    m2 = np.abs(r33 + 1) < tol            # tilt ~ 180
    mg = ~m1 & ~m2
    narot[m1] = np.degrees(np.arctan2(R[m1, 1, 0], R[m1, 0, 0]))
    tilt[m2] = 180.0
    narot[m2] = np.degrees(np.arctan2(R[m2, 1, 0], R[m2, 0, 0]))
    tdrot[mg] = np.degrees(np.arctan2(R[mg, 2, 0], R[mg, 2, 1]))
    tilt[mg] = np.degrees(np.arccos(r33[mg]))
    narot[mg] = np.degrees(np.arctan2(R[mg, 0, 2], -R[mg, 1, 2]))
    out = np.column_stack([tdrot, tilt, narot])
    return out[0] if single else out


def dynamo_euler2matrix(tdrot, tilt, narot):
    """Inverse of :func:`dynamo_matrix2euler`: R = Rz(narot) Rx(tilt) Rz(tdrot)."""
    return (rotation_matrix_z(narot) @ rotation_matrix_x(tilt)
            @ rotation_matrix_z(tdrot))


def tom_to_dynamo_eulers(motl):
    """Return (N,3) Dynamo table Euler angles [tdrot, tilt, narot] in degrees.

    Matches ArtiaX's DynamoEulerRotation: Dynamo stores the *object* orientation
    directly (no inversion), so the table angles are dynamo_matrix2euler(R) of
    the particle rotation R = Rz(psi)Rx(theta)Rz(phi).
    """
    return dynamo_matrix2euler(tom_rotation_matrices(motl))


def relion_eulers_from_matrix(R):
    """Extract RELION (rot, tilt, psi) degrees from object-orientation matrix R.

    Vectorised port of ArtiaX's RELIONEulerRotation (ZYZ, invert_dir=True), so
    that RELION's stored angles describe the inverse of the particle's object
    orientation, matching how RELION/Warp interpret rlnAngleRot/Tilt/Psi.
    Accepts (3,3) or (N,3,3); returns (3,) or (N,3).
    """
    R = np.asarray(R, dtype=np.float64)
    single = (R.ndim == 2)
    if single:
        R = R[None, :, :]
    eps = 1e-6
    m02, m12, m22 = R[:, 0, 2], R[:, 1, 2], R[:, 2, 2]
    m20, m21 = R[:, 2, 0], R[:, 2, 1]
    m00, m10 = R[:, 0, 0], R[:, 1, 0]
    abs_sb = np.sqrt(m02 * m02 + m12 * m12)
    nd = abs_sb > eps                      # non-degenerate (tilt not 0/180)

    rot = np.zeros(R.shape[0])
    tilt = np.zeros(R.shape[0])
    psi = np.zeros(R.shape[0])

    # sign of sin(tilt), following ArtiaX._sign_rot2
    rot3 = np.arctan2(m12, -m02)
    s3 = np.sin(rot3)
    with np.errstate(divide="ignore", invalid="ignore"):
        sign_small = np.sign(-m02 / np.cos(rot3))
    sign_sb = np.where(np.abs(s3) < eps, sign_small,
                       np.where(s3 > 0, np.sign(m12), -np.sign(m12)))

    rot[nd] = np.arctan2(m21[nd], m20[nd])
    tilt[nd] = np.arctan2(sign_sb[nd] * abs_sb[nd], m22[nd])
    psi[nd] = np.arctan2(m12[nd], -m02[nd])
    # degenerate: tilt 0 or 180, fold spin into psi
    dpos = (~nd) & (m22 >= 0)
    dneg = (~nd) & (m22 < 0)
    tilt[dneg] = np.pi
    psi[dpos] = np.arctan2(-m10[dpos], m00[dpos])
    psi[dneg] = np.arctan2(m10[dneg], -m00[dneg])

    out = np.degrees(np.column_stack([rot, tilt, psi]))
    return out[0] if single else out


def tom_to_relion_eulers(motl):
    """Return (N,3) RELION ZYZ Euler angles [rot, tilt, psi] in degrees."""
    return relion_eulers_from_matrix(tom_rotation_matrices(motl))


# ---------------------------------------------------------------------------
# Dynamo .tbl writer (35-column format, ported from write_dynamo_table)
# ---------------------------------------------------------------------------
# 1-indexed integer columns that must be written as integers:
_DYNAMO_INT_COLS = {0, 1, 2, 11, 12, 19, 20, 21, 30, 31, 33, 34}


def write_dynamo_tbl(path, motl, pos_vox1, shifts_vox, tomo_id=1):
    """Write a Dynamo particle table (.tbl).

    ``pos_vox1`` : (N,3) 1-indexed integer voxel positions (cols 24-26).
    ``shifts_vox``: (N,3) sub-voxel shifts (cols 4-6).
    ``tomo_id``  : tomogram index for col 20 (matches the .vll line number).
    """
    n = motl.shape[1]
    eul = tom_to_dynamo_eulers(motl)
    table = np.zeros((n, 35))
    tag = np.arange(1, n + 1, dtype=float)
    table[:, 0] = tag                       # col 1:  particle tag
    table[:, 3:6] = shifts_vox              # col 4-6: dx dy dz
    table[:, 6:9] = eul                     # col 7-9: tdrot tilt narot
    table[:, 9] = motl[ROW_CCC, :]          # col 10: cc
    table[:, 12] = 1.0                      # col 13: Fourier sampling
    table[:, 13] = -60.0                    # col 14: tilt min
    table[:, 14] = 60.0                     # col 15: tilt max
    table[:, 19] = float(tomo_id)           # col 20: tomogram
    table[:, 20] = tag                      # col 21: region (= tag)
    table[:, 21] = 1.0                      # col 22
    table[:, 23:26] = pos_vox1             # col 24-26: x y z
    table[:, 34] = 1.0                      # col 35
    lines = []
    for row in table:
        parts = [str(int(round(v))) if j in _DYNAMO_INT_COLS else "%g" % v
                 for j, v in enumerate(row)]
        lines.append(" ".join(parts))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# RELION star writers
# ---------------------------------------------------------------------------
def write_relion5_star(path, motl, centered_A, tomo_name="tomo",
                       manifold=None, subset=None):
    """Write a RELION 5.1 particles STAR file (centered Angstrom coordinates).

    Orientation goes in rlnTomoSubtomogram{Rot,Tilt,Psi}; rlnAngle* start at 0.
    """
    n = motl.shape[1]
    eul = tom_to_relion_eulers(motl)
    if manifold is None:
        manifold = motl[1, :].astype(int)
    if subset is None:
        subset = (np.arange(n) % 2) + 1
    header = (
        "# Generated by Geopickr\n\ndata_particles\n\nloop_\n"
        "_rlnTomoName #1\n_rlnTomoParticleName #2\n_rlnRandomSubset #3\n"
        "_rlnCenteredCoordinateXAngst #4\n_rlnCenteredCoordinateYAngst #5\n"
        "_rlnCenteredCoordinateZAngst #6\n_rlnOriginXAngst #7\n"
        "_rlnOriginYAngst #8\n_rlnOriginZAngst #9\n_rlnAngleRot #10\n"
        "_rlnAngleTilt #11\n_rlnAnglePsi #12\n_rlnTomoSubtomogramRot #13\n"
        "_rlnTomoSubtomogramTilt #14\n_rlnTomoSubtomogramPsi #15\n"
        "_rlnTomoManifoldIndex #16\n"
    )
    lines = []
    for i in range(n):
        x, y, z = centered_A[i]
        rot, tilt, psi = eul[i]
        lines.append(
            "%s  %s/%06d  %d  %g  %g  %g  0.0  0.0  0.0  0.0  0.0  0.0  "
            "%g  %g  %g  %d"
            % (tomo_name, tomo_name, i + 1, int(subset[i]), x, y, z,
               rot, tilt, psi, int(manifold[i])))
    with open(path, "w") as f:
        f.write(header + "\n".join(lines) + "\n")


def write_relion3_star(path, motl, pos_vox0, tomo_name="tomo"):
    """Write a RELION 3/4-style tomo particles STAR file (pixel coordinates)."""
    n = motl.shape[1]
    eul = tom_to_relion_eulers(motl)
    header = (
        "# Generated by Geopickr\n\ndata_particles\n\nloop_\n"
        "_rlnTomoName #1\n_rlnCoordinateX #2\n_rlnCoordinateY #3\n"
        "_rlnCoordinateZ #4\n_rlnAngleRot #5\n_rlnAngleTilt #6\n_rlnAnglePsi #7\n"
    )
    lines = []
    for i in range(n):
        x, y, z = pos_vox0[i]
        rot, tilt, psi = eul[i]
        lines.append("%s  %g  %g  %g  %g  %g  %g"
                     % (tomo_name, x, y, z, rot, tilt, psi))
    with open(path, "w") as f:
        f.write(header + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Dynamo .vll volume-list integration (ported from vll_to_dynamo_tables.py)
# ---------------------------------------------------------------------------
def _is_vll_tomo_line(line):
    s = line.strip()
    return bool(s) and not s.startswith(("#", "$", "*", ">"))


def read_vll(path):
    with open(path) as f:
        return f.readlines()


def vll_tomo_index(lines, tomo_substr):
    """Return the 1-based index of the .vll tomogram line containing ``tomo_substr``.

    Returns None if not found.
    """
    idx = 0
    for line in lines:
        if _is_vll_tomo_line(line):
            idx += 1
            if tomo_substr and tomo_substr in line:
                return idx
    return None


def append_vll_table_ref(vll_path, tbl_path, tomo_substr=None, backup=True):
    """Append a '> tbl_path' reference into a .vll after the matching tomogram block.

    If ``tomo_substr`` is None or not found, the reference is appended to the
    last tomogram block. Writes a .vll.bak backup first when ``backup`` is True.
    Returns the 1-based tomogram index the reference was attached to (or None).
    """
    lines = read_vll(vll_path)
    target = vll_tomo_index(lines, tomo_substr) if tomo_substr else None
    if target is None:
        # default: last tomogram block
        total = sum(1 for l in lines if _is_vll_tomo_line(l))
        target = total or None
    if target is None:
        return None

    result = []
    idx = 0
    i = 0
    ref = "> %s\n" % tbl_path
    while i < len(lines):
        line = lines[i]
        if not _is_vll_tomo_line(line):
            result.append(line)
            i += 1
            continue
        idx += 1
        block = [line]
        i += 1
        while i < len(lines) and not _is_vll_tomo_line(lines[i]):
            block.append(lines[i])
            i += 1
        result.extend(block)
        if idx == target:
            existing = {l.strip()[1:].strip() for l in block
                        if l.strip().startswith(">")}
            if str(tbl_path) not in existing:
                result.append(ref)

    if backup:
        import shutil
        shutil.copy2(vll_path, str(vll_path) + ".bak")
    with open(vll_path, "w") as f:
        f.writelines(result)
    return target
