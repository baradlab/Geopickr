# vim: set expandtab ts=4 sw=4:
"""Geometric particle picking.

Generates 20xN TOM/AV3 motive lists by sampling evenly-spaced particle
positions and orientations on:

  * sphere   - port of the original Pick Particle ``Sampling_sphere``
  * tube     - port of ``Sampling_tube`` (particles on the tube surface)
  * filament - NEW: particles on the spline centerline, Z along the tangent
  * surface  - NEW: area-weighted, Poisson-thinned sampling of a mesh

The sphere/tube ports reproduce the maths of K. Qu's Chimera plugin; filament
and surface are new derivations sharing the same normal->Euler convention
(:func:`chimerax.geopickr.motivelist.normal_to_zxz`).
"""

import numpy as np

from . import motivelist as ml


# ---------------------------------------------------------------------------
# Spherical <-> Cartesian helpers (match the original plugin)
# ---------------------------------------------------------------------------
def _sph2cart(az, el, r):
    rc = r * np.cos(el)
    return rc * np.cos(az), rc * np.sin(az), r * np.sin(el)


def _cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    return np.arctan2(y, x), np.arctan2(z, hxy), np.hypot(hxy, z)


def _finalize(coords, random_phi, tomo_id, offset=0.0):
    """Common tail applied to every sampler's raw (20,N) coordinate list.

    ``offset`` moves each particle along its own +Z axis (the surface normal for
    sphere/tube/surface, the axis tangent for filament) by that many voxels, so
    the offset is baked into the coordinates.  Positive = along +Z (outward).
    """
    if coords.shape[1] == 0:
        return coords
    m = coords
    if offset:
        # +Z of each particle in global coords is column 2 of its rotation.
        zdir = ml.rotation_matrices_zxz(m[16:19, :])[:, :, 2].T   # (3, N)
        m[7:10, :] = m[7:10, :] + float(offset) * zdir
    # split absolute position into integer voxel (rows 8-10) + shift (rows 11-13)
    m[10:13, :] = m[7:10, :] - np.round(m[7:10, :])
    m[7:10, :] = np.round(m[7:10, :])
    if random_phi:
        m[16, :] = np.random.uniform(0.0, 1.0, m.shape[1]) * 360.0
    m[17, :] = np.mod(m[17, :], 360.0)
    m[3, :] = np.arange(1, m.shape[1] + 1)     # particle number
    m[19, :] += 1                              # class -> start at 1
    m[4, :] = tomo_id                          # running tomogram number
    m[6, :] = tomo_id                          # tomogram number
    return m


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------
def sample_sphere(centers, radii, t_spacing, random_phi=True, tomo_id=0,
                  set_ids=None, offset=0.0):
    """Sample particles on one or more spheres.

    ``centers`` is (K,3); ``radii`` a scalar or length-K; spacing in voxels.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    k = len(centers)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (k,))
    if set_ids is None:
        set_ids = np.zeros(k, dtype=np.float64)
    t = float(t_spacing)
    cols = []
    for nsph in range(k):
        c = centers[nsph]
        r = float(radii[nsph])
        if r <= 0 or t <= 0:
            continue
        latitudes = int(np.ceil(np.pi * r / t))
        for ele in range(latitudes + 1):
            elevation = (ele / latitudes - 0.5) * np.pi
            longitudes = int(np.ceil(2 * np.pi * r * np.cos(elevation) / t))
            theta = 90.0 - np.rad2deg(elevation)
            for azi in range(longitudes):
                azimuth = azi * 2 * np.pi / longitudes
                x, y, z = _sph2cart(azimuth, elevation, r)
                psi = np.rad2deg(azimuth) - 270.0
                col = np.zeros(20)
                col[1] = set_ids[nsph]
                col[2] = r
                col[5] = nsph + 1
                col[7] = x + c[0]
                col[8] = y + c[1]
                col[9] = z + c[2]
                col[17] = psi
                col[18] = theta
                cols.append(col)
    coords = (np.array(cols).T if cols else np.zeros((20, 0)))
    return _finalize(coords, random_phi, tomo_id, offset=offset)


# ---------------------------------------------------------------------------
# Tube / filament shared spline resampling
# ---------------------------------------------------------------------------
def _dense_spline(axis_pts, spacing):
    pts = np.asarray(axis_pts, dtype=np.float64)
    chord = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    n_dense = max(50, int(chord / max(spacing, 1e-6) * 20) + 1)
    return ml.natural_cubic_spline(pts, n_dense)


def _resample_arclength(pos, tan, spacing):
    """Return (positions, unit_tangents) spaced ~``spacing`` along arc length."""
    if len(pos) < 2 or spacing <= 0:
        return pos[:1], tan[:1]
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total == 0:
        return pos[:1], tan[:1]
    targets = np.arange(0.0, total + 1e-9, spacing)
    idx = np.clip(np.searchsorted(s, targets), 0, len(pos) - 1)
    return pos[idx], tan[idx]


# ---------------------------------------------------------------------------
# Tube (particles on the tube surface at radius r)
# ---------------------------------------------------------------------------
def sample_tube(axis_list, radius, t_spacing, a_spacing=0.0,
                random_phi=False, tomo_id=0, set_ids=None, offset=0.0):
    """Sample particles on the surface of one or more tubes.

    ``axis_list`` is a list of (M,3) point arrays (>=2 per tube).
    """
    t = float(t_spacing)
    a = float(a_spacing) if a_spacing and a_spacing > 0 else t
    radii = np.broadcast_to(np.asarray(radius, dtype=np.float64),
                            (len(axis_list),))
    cols = []
    for ntb, axis in enumerate(axis_list):
        axis = np.asarray(axis, dtype=np.float64)
        r = float(radii[ntb])
        if len(axis) < 2 or r <= 0 or t <= 0:
            continue
        sid = set_ids[ntb] if set_ids is not None else ntb
        pos, tan = _dense_spline(axis, a)
        jpos, jtan = _resample_arclength(pos, tan, a)
        nlong = int(np.ceil(2 * np.pi * r / t))
        l_spacing = 2 * np.pi / nlong
        longitudes = np.rad2deg(np.arange(0.0, 2 * np.pi, l_spacing))
        shifts = np.array([r, 0.0, 0.0])
        for p, tdir in zip(jpos, jtan):
            az, el, _ = _cart2sph(tdir[0], tdir[1], tdir[2])
            az_d, el_d = np.rad2deg(az), np.rad2deg(el)
            for angle in longitudes:
                rotzyz = (ml.rotation_matrix_z(az_d)
                          @ ml.rotation_matrix_y(-el_d - 90.0)
                          @ ml.rotation_matrix_z(angle))
                rshift = rotzyz @ shifts
                psi = np.rad2deg(np.arctan2(rshift[1], rshift[0])) + 90.0
                theta = np.rad2deg(np.arctan2(
                    np.hypot(rshift[0], rshift[1]), rshift[2]))
                rotzxz = ml.rotation_matrix_zxz([0.0, psi, theta])
                phi_vec = rotzxz.T @ tdir
                phi = np.rad2deg(np.arctan2(phi_vec[1], phi_vec[0]))
                col = np.zeros(20)
                col[1] = sid
                col[2] = r
                col[5] = ntb + 1
                col[7] = rshift[0] + p[0]
                col[8] = rshift[1] + p[1]
                col[9] = rshift[2] + p[2]
                col[16] = phi
                col[17] = psi
                col[18] = theta
                cols.append(col)
    coords = (np.array(cols).T if cols else np.zeros((20, 0)))
    return _finalize(coords, random_phi, tomo_id, offset=offset)


# ---------------------------------------------------------------------------
# Filament (particles on the spline centerline, Z along the tangent)
# ---------------------------------------------------------------------------
def sample_filament(axis_list, a_spacing, twist_deg=0.0, random_phi=False,
                    tomo_id=0, set_ids=None, offset=0.0):
    """Sample particles along the centerline of one or more filaments."""
    a = float(a_spacing)
    cols = []
    for nfl, axis in enumerate(axis_list):
        axis = np.asarray(axis, dtype=np.float64)
        if len(axis) < 2 or a <= 0:
            continue
        sid = set_ids[nfl] if set_ids is not None else nfl
        pos, tan = _dense_spline(axis, a)
        jpos, jtan = _resample_arclength(pos, tan, a)
        for i, (p, tdir) in enumerate(zip(jpos, jtan)):
            psi, theta = ml.normal_to_zxz(tdir)
            col = np.zeros(20)
            col[1] = sid
            col[5] = nfl + 1
            col[7], col[8], col[9] = p
            col[16] = i * float(twist_deg)
            col[17] = psi
            col[18] = theta
            cols.append(col)
    coords = (np.array(cols).T if cols else np.zeros((20, 0)))
    return _finalize(coords, random_phi, tomo_id, offset=offset)


# ---------------------------------------------------------------------------
# Surface (area-weighted, Poisson-disk thinned mesh sampling)
# ---------------------------------------------------------------------------
# Safety cap so an over-fine spacing on a huge surface can't allocate/blow up.
MAX_SURFACE_PARTICLES = 200000


def sample_surface(vertices, triangles, normals=None, t_spacing=10.0,
                   random_phi=True, tomo_id=0, oversample=5, offset=0.0):
    """Sample particles roughly ``t_spacing`` apart over a triangle mesh.

    Orientations align the object +Z with the (interpolated) surface normal.
    """
    v = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    t = float(t_spacing)
    if len(tri) == 0 or t <= 0:
        return np.zeros((20, 0))

    v0, v1, v2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    face_area = 0.5 * np.linalg.norm(face_n, axis=1)
    total_area = float(face_area.sum())
    if total_area <= 0:
        return np.zeros((20, 0))

    target = max(1, int(round(total_area / (t * t))))
    target = min(target, MAX_SURFACE_PARTICLES)
    n_cand = min(max(target * int(oversample), target + 1),
                 MAX_SURFACE_PARTICLES * int(oversample))

    probs = face_area / total_area
    rng = np.random
    fidx = rng.choice(len(tri), size=n_cand, p=probs)
    r1 = np.sqrt(rng.uniform(0, 1, n_cand))
    r2 = rng.uniform(0, 1, n_cand)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    pts = (a[:, None] * v0[fidx] + b[:, None] * v1[fidx] + c[:, None] * v2[fidx])

    if normals is not None:
        n = np.asarray(normals, dtype=np.float64)
        nrm = (a[:, None] * n[tri[fidx, 0]] + b[:, None] * n[tri[fidx, 1]]
               + c[:, None] * n[tri[fidx, 2]])
    else:
        nrm = face_n[fidx]
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    nrm = nrm / nlen

    keep = _poisson_thin(pts, t, target)

    cols = []
    for i in keep:
        psi, theta = ml.normal_to_zxz(nrm[i])
        col = np.zeros(20)
        col[5] = 1
        col[7], col[8], col[9] = pts[i]
        col[17] = psi
        col[18] = theta
        cols.append(col)
    coords = (np.array(cols).T if cols else np.zeros((20, 0)))
    return _finalize(coords, random_phi, tomo_id, offset=offset)


def _poisson_thin(pts, min_dist, target=None):
    """Greedy Poisson-disk thinning; return indices kept (>= min_dist apart).

    Builds one KD-tree over all candidates and, in random order, accepts a
    point then blocks every candidate within ``min_dist`` of it.  This is
    roughly O(N log N) rather than rebuilding a tree per point.
    """
    from scipy.spatial import cKDTree
    n = len(pts)
    if n == 0:
        return []
    tree = cKDTree(pts)
    alive = np.ones(n, dtype=bool)
    kept = []
    for i in np.random.permutation(n):
        if not alive[i]:
            continue
        kept.append(i)
        alive[tree.query_ball_point(pts[i], min_dist)] = False
    return kept


# ---------------------------------------------------------------------------
# Orchestrator used by the command and GUI
# ---------------------------------------------------------------------------
def _markerset_coords(ms):
    return np.asarray(ms.atoms.coords, dtype=np.float64)


def pick(session, *, style, marker_models=None, surface_model=None,
         radius=20.0, tangential=0.0, axial=0.0, twist=0.0,
         random_phi=None, tomo_id=0, offset=0.0):
    """High-level dispatch returning a (20,N) motive list (uniform radius)."""
    style = style.lower()
    marker_models = marker_models or []

    if style == "sphere":
        centers, set_ids = [], []
        for sid, ms in enumerate(marker_models):
            for c in _markerset_coords(ms):
                centers.append(c)
                set_ids.append(sid)
        rp = True if random_phi is None else random_phi
        return sample_sphere(np.array(centers), radius, tangential,
                             random_phi=rp, tomo_id=tomo_id,
                             set_ids=np.array(set_ids), offset=offset)

    if style == "tube":
        axis_list = [_markerset_coords(ms) for ms in marker_models]
        rp = False if random_phi is None else random_phi
        return sample_tube(axis_list, radius, tangential, axial,
                           random_phi=rp, tomo_id=tomo_id, offset=offset)

    if style == "filament":
        axis_list = [_markerset_coords(ms) for ms in marker_models]
        rp = False if random_phi is None else random_phi
        return sample_filament(axis_list, axial or tangential, twist_deg=twist,
                              random_phi=rp, tomo_id=tomo_id, offset=offset)

    if style == "surface":
        if surface_model is None:
            from chimerax.core.errors import UserError
            raise UserError("Surface picking needs a surface model (onSurface).")
        v = surface_model.vertices
        tri = surface_model.triangles
        nrm = getattr(surface_model, "normals", None)
        rp = True if random_phi is None else random_phi
        return sample_surface(v, tri, nrm, tangential, random_phi=rp,
                             tomo_id=tomo_id, offset=offset)

    from chimerax.core.errors import UserError
    raise UserError("Unknown picking style: %s" % style)
