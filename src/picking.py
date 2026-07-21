# vim: set expandtab ts=4 sw=4:
"""Geometric particle picking.

Generates 20xN TOM/AV3 motive lists by sampling evenly-spaced particle
positions and orientations on:

  * sphere   - port of the original Pick Particle ``Sampling_sphere``
  * tube     - port of ``Sampling_tube`` (particles on the tube surface)
  * filament - NEW: particles on the spline centerline, Z along the tangent
  * surface  - NEW: restricted-Lloyd / CVT sampling of a mesh (even density)

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
# Surface (restricted-Lloyd / CVT mesh sampling)
# ---------------------------------------------------------------------------
# Safety cap so an over-fine spacing on a huge surface can't allocate/blow up.
MAX_SURFACE_PARTICLES = 200000
# Cap on the dense point cloud used as the surface proxy for the CVT, so a very
# fine spacing on a big mesh can't blow up memory / KD-tree time.
_SURFACE_CAND_CAP = 400000
# Restricted-Lloyd relaxation iterations (a few dozen is plenty to reach a
# near-hexagonal centroidal Voronoi tessellation on the discrete surface).
_LLOYD_ITERS = 50
# The CVT uses a straight-line (chord) metric, so on curved regions it leaves a
# small tail of pairs closer than the ideal spacing.  A short min-distance
# repulsion pass afterwards pushes those apart (reprojecting onto the surface)
# without disturbing the even layout.  The floor is a fraction of the tangential
# spacing (the ideal near-hexagonal nearest-neighbor distance is ~1.07x it).
_REPEL_ITERS = 15
_MIN_DIST_FRAC = 0.75


def _area_weighted_cloud(v0, v1, v2, face_n, probs, n, normals, tri, fidx_rng):
    """Draw ``n`` area-weighted points (with unit normals) over the mesh."""
    fidx = fidx_rng.choice(len(probs), size=n, p=probs)
    r1 = np.sqrt(fidx_rng.uniform(0, 1, n))
    r2 = fidx_rng.uniform(0, 1, n)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    pts = a[:, None] * v0[fidx] + b[:, None] * v1[fidx] + c[:, None] * v2[fidx]
    if normals is not None:
        nn = np.asarray(normals, dtype=np.float64)
        nrm = (a[:, None] * nn[tri[fidx, 0]] + b[:, None] * nn[tri[fidx, 1]]
               + c[:, None] * nn[tri[fidx, 2]])
    else:
        nrm = face_n[fidx].astype(np.float64)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    return pts, nrm / nlen, fidx


def _restricted_lloyd(cloud, cloud_tree, target, iters, rng):
    """Restricted-Lloyd / CVT relaxation over a dense surface point cloud.

    The cloud approximates the surface and its area measure.  Starting from
    ``target`` random seeds, each iteration (a) assigns every cloud point to its
    nearest seed (a Voronoi cell) and (b) moves each seed to its cell's centroid.
    The seeds relax into an even, near-hexagonal (blue-noise) layout.  Seeds
    float slightly off a curved surface as they average, so at the end each is
    projected back onto the surface (nearest cloud point).  Returns the indices
    (into ``cloud``) of the final, de-duplicated seeds.
    """
    from scipy.spatial import cKDTree
    n = len(cloud)
    if target >= n:
        return np.arange(n)
    seeds = cloud[rng.choice(n, size=target, replace=False)].copy()
    for _ in range(iters):
        _, labels = cKDTree(seeds).query(cloud)
        counts = np.bincount(labels, minlength=target)
        centroids = np.zeros((target, 3))
        np.add.at(centroids, labels, cloud)
        nonempty = counts > 0
        centroids[nonempty] /= counts[nonempty, None]
        centroids[~nonempty] = seeds[~nonempty]     # keep empty cells put
        seeds = centroids
    # Project the relaxed seeds back onto the discrete surface, keeping them
    # distinct (two seeds occasionally snap to the same cloud point).
    _, snapped = cloud_tree.query(seeds)
    return _dedup_indices(snapped, cloud, cloud_tree, rng)


def _repel_on_surface(idx, cloud, cloud_tree, min_dist, iters, rng):
    """Push apart seed pairs closer than ``min_dist``, staying on the surface.

    The CVT's chord metric leaves a few pairs closer than the ideal spacing on
    curved regions.  Each iteration displaces both members of every too-close
    pair symmetrically along their separation, then reprojects onto the surface
    (nearest cloud point).  Converges in a few iterations; returns updated,
    de-duplicated cloud indices.
    """
    from scipy.spatial import cKDTree
    idx = np.asarray(idx, dtype=np.int64)
    seeds = cloud[idx].copy()
    for _ in range(iters):
        pairs = np.asarray(list(cKDTree(seeds).query_pairs(min_dist)),
                           dtype=np.int64)
        if len(pairs) == 0:
            break
        d = seeds[pairs[:, 1]] - seeds[pairs[:, 0]]
        length = np.linalg.norm(d, axis=1, keepdims=True)
        length[length == 0] = 1.0
        push = (min_dist - length) * 0.5 * d / length
        disp = np.zeros_like(seeds)
        np.add.at(disp, pairs[:, 0], -push)
        np.add.at(disp, pairs[:, 1], push)
        _, idx = cloud_tree.query(seeds + disp)     # reproject to the surface
        seeds = cloud[idx]
    return _dedup_indices(idx, cloud, cloud_tree, rng)


def _dedup_indices(idx, cloud, cloud_tree, rng):
    """Make ``idx`` distinct, resolving collisions to nearby unused cloud points."""
    n = len(cloud)
    used = set()
    out = np.array(idx, dtype=np.int64)
    for k in range(len(out)):
        j = int(out[k])
        if j not in used:
            used.add(j)
            continue
        # Try progressively larger neighborhoods for an unused cloud point.
        placed = False
        for kk in (8, 32, 128):
            cand = np.atleast_1d(cloud_tree.query(cloud[j], k=kk)[1])
            for c in cand:
                c = int(c)
                if c not in used:
                    used.add(c)
                    out[k] = c
                    placed = True
                    break
            if placed:
                break
        if not placed:
            r = int(rng.integers(0, n))
            while r in used:
                r = int(rng.integers(0, n))
            used.add(r)
            out[k] = r
    return out


def _jitter_tangent(pts, nrm, jitter, rng):
    """Perturb each point by a random vector in its tangent plane.

    ``jitter`` is the maximum displacement (same units as the mesh, i.e. voxels).
    Points stay (locally) on the surface because the displacement is orthogonal
    to the normal; the normal/orientation is left unchanged.
    """
    n = len(pts)
    if n == 0 or jitter <= 0:
        return pts
    # A stable tangent basis per point: cross the normal with whichever global
    # axis it is least aligned to, then complete the frame.
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    flip = np.abs(nrm[:, 2]) > 0.9
    ref[flip] = np.array([1.0, 0.0, 0.0])
    u = np.cross(nrm, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(nrm, u)
    rad = jitter * np.sqrt(rng.uniform(0, 1, n))       # uniform over the disk
    ang = rng.uniform(0, 2 * np.pi, n)
    return pts + (rad * np.cos(ang))[:, None] * u + (rad * np.sin(ang))[:, None] * w


def sample_surface(vertices, triangles, normals=None, t_spacing=10.0,
                   random_phi=True, tomo_id=0, oversample=30, offset=0.0,
                   jitter=0.0, component=None, seed=None):
    """Sample particles roughly ``t_spacing`` apart over a triangle mesh.

    Draws a dense area-weighted point cloud as a surface proxy, then runs a
    restricted Lloyd / centroidal-Voronoi relaxation to spread ``area / t^2``
    seeds into an even, near-hexagonal density, followed by a short min-distance
    repulsion pass that clears the close-pair tail the chord-metric CVT leaves on
    curved regions.  Orientations align the object +Z with the (interpolated)
    surface normal.  An optional ``jitter`` (voxels) applies a small random
    tangential perturbation afterwards, so users who want to break up the regular
    lattice can.

    ``component`` is an optional per-triangle integer label (length = number of
    triangles, e.g. a VTP ``component_number``); when given, each particle's
    ``_object`` field (row 5) is set to the component of the face it was sampled
    from, so downstream export can keep whole components together (half-set
    splitting).
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
    # Dense cloud approximating the surface; more points -> smoother centroids.
    n_cloud = min(max(target * int(oversample), target + 1), _SURFACE_CAND_CAP)

    rng = np.random.default_rng(seed)
    probs = face_area / total_area
    cloud, cloud_nrm, cloud_fidx = _area_weighted_cloud(
        v0, v1, v2, face_n, probs, n_cloud, normals, tri, rng)
    cloud_comp = None
    if component is not None:
        cloud_comp = np.asarray(component, dtype=np.int64)[cloud_fidx]

    from scipy.spatial import cKDTree
    cloud_tree = cKDTree(cloud)
    keep = _restricted_lloyd(cloud, cloud_tree, target, _LLOYD_ITERS, rng)
    keep = _repel_on_surface(keep, cloud, cloud_tree, _MIN_DIST_FRAC * t,
                             _REPEL_ITERS, rng)
    pts = cloud[keep]
    nrm = cloud_nrm[keep]
    comp = cloud_comp[keep] if cloud_comp is not None else None
    if jitter > 0:
        pts = _jitter_tangent(pts, nrm, float(jitter), rng)

    cols = []
    for i in range(len(pts)):
        psi, theta = ml.normal_to_zxz(nrm[i])
        col = np.zeros(20)
        col[5] = int(comp[i]) if comp is not None else 1
        col[7], col[8], col[9] = pts[i]
        col[17] = psi
        col[18] = theta
        cols.append(col)
    coords = (np.array(cols).T if cols else np.zeros((20, 0)))
    return _finalize(coords, random_phi, tomo_id, offset=offset)


# ---------------------------------------------------------------------------
# Orchestrator used by the command and GUI
# ---------------------------------------------------------------------------
def surface_component_faces(surface_model):
    """Per-triangle ``component_number`` for a VTP surface, or None.

    MorphometricsX loads quantified ``.vtp`` meshes as a ChimeraX ``Surface``
    carrying a ``.vtp`` object with ``cell_data`` / ``point_data`` arrays (from
    surface_morphometrics).  When a ``component_number`` array is present we
    return it aligned to ``surface_model.triangles`` (VTP cell order), so each
    picked particle can be tagged with the connected surface component it lands
    on.  Duck-typed so Geopickr never hard-depends on MorphometricsX.
    """
    vtp = getattr(surface_model, "vtp", None)
    if vtp is None:
        return None
    cell = getattr(vtp, "cell_data", None) or {}
    arr = cell.get("component_number")
    if arr is not None:                         # per-triangle: direct
        arr = np.asarray(arr).reshape(-1)
        return arr.astype(np.int64) if len(arr) else None
    point = getattr(vtp, "point_data", None) or {}
    parr = point.get("component_number")
    if parr is not None:                        # per-vertex: label by 1st vertex
        parr = np.asarray(parr).reshape(-1)
        tris = np.asarray(getattr(vtp, "triangles"))
        return parr.astype(np.int64)[tris[:, 0]]
    return None


def has_multiple_objects(motl):
    """True if the motl spans >= 2 distinct ``_object`` (row 5) ids.

    Every sampler tags particles with a per-object id (one per sphere, tube,
    filament, or surface component).  When a pick produced more than one object,
    STOPGAP export keeps whole objects together in the gold-standard halves
    (instead of alternating particle-by-particle), so independent objects are
    never split across the two halves.
    """
    m = np.asarray(motl)
    if m.ndim != 2 or m.shape[1] == 0:
        return False
    return int(np.unique(m[5, :]).size) >= 2


def _markerset_coords(ms):
    return np.asarray(ms.atoms.coords, dtype=np.float64)


def pick(session, *, style, marker_models=None, surface_model=None,
         radius=20.0, tangential=0.0, axial=0.0, twist=0.0,
         random_phi=None, tomo_id=0, offset=0.0, jitter=0.0):
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
        comp = surface_component_faces(surface_model)
        rp = True if random_phi is None else random_phi
        return sample_surface(v, tri, nrm, tangential, random_phi=rp,
                             tomo_id=tomo_id, offset=offset, jitter=jitter,
                             component=comp)

    from chimerax.core.errors import UserError
    raise UserError("Unknown picking style: %s" % style)
