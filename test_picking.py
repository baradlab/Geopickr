"""Headless sampling tests. Run: ChimeraX --nogui --exit --script test_picking.py"""
import numpy as np
from chimerax.geopickr import motivelist as ml
from chimerax.geopickr import picking as pk

np.random.seed(0)

# ---- sphere ------------------------------------------------------------
r, t = 50.0, 10.0
center = np.array([[100.0, 200.0, 300.0]])
m = pk.sample_sphere(center, r, t, random_phi=False, tomo_id=7)
n = m.shape[1]
assert n > 0
# every particle lies on the sphere of radius r about the center
pos = m[7:10, :] + m[10:13, :]        # integer + shift = absolute
d = np.linalg.norm(pos - center.T, axis=0)
assert np.allclose(d, r, atol=1.0), (d.min(), d.max())
# expected count from the latitude/longitude formula
lat = int(np.ceil(np.pi * r / t))
exp = 0
for ele in range(lat + 1):
    el = (ele / lat - 0.5) * np.pi
    exp += int(np.ceil(2 * np.pi * r * np.cos(el) / t))
assert n == exp, (n, exp)
# orientation: object +Z should map to the radial (outward) normal
i = n // 2
R = ml.rotation_matrix_zxz([m[16, i], m[17, i], m[18, i]])
zaxis = R @ np.array([0, 0, 1.0])
radial = (pos[:, i] - center[0]) / r
assert np.allclose(zaxis, radial, atol=0.05), (zaxis, radial)
assert (m[4, :] == 7).all() and (m[6, :] == 7).all()   # tomo id stamped
assert (m[19, :] == 1).all()                            # class -> 1
print("OK sphere: %d particles on r=%g, normals radial" % (n, r))

# ---- filament ----------------------------------------------------------
axis = np.array([[0, 0, 0], [0, 0, 100.0], [0, 0, 200.0]])
mf = pk.sample_filament([axis], a_spacing=10.0, twist_deg=30.0,
                        random_phi=False, tomo_id=0)
nf = mf.shape[1]
posf = mf[7:10, :] + mf[10:13, :]
# points lie on the z-axis (x,y ~ 0)
assert np.allclose(posf[0], 0, atol=1e-6) and np.allclose(posf[1], 0, atol=1e-6)
# axial spacing ~ 10 in z
dz = np.diff(np.sort(posf[2]))
assert np.allclose(dz, 10.0, atol=1.0), dz
# Z orientation along filament tangent (+z)
R = ml.rotation_matrix_zxz([mf[16, 1], mf[17, 1], mf[18, 1]])
zf = R @ np.array([0, 0, 1.0])
assert np.allclose(zf, [0, 0, 1], atol=0.05), zf
# helical twist accumulates
assert abs(mf[16, 2] - mf[16, 1] - 30.0) < 1e-6
print("OK filament: %d particles on axis, twist accumulates" % nf)

# ---- tube --------------------------------------------------------------
mt = pk.sample_tube([axis], radius=20.0, t_spacing=10.0, a_spacing=10.0,
                    random_phi=False, tomo_id=0)
nt = mt.shape[1]
post = mt[7:10, :] + mt[10:13, :]
# radial distance from the z-axis ~ radius
rad = np.hypot(post[0], post[1])
assert np.allclose(rad, 20.0, atol=1.5), (rad.min(), rad.max())
print("OK tube: %d particles at radius 20 from axis" % nt)

# ---- surface (unit-ish icosphere via two triangles plane is too flat; use a box face grid) ----
# Build a simple 100x100 planar mesh in z=0 with +z normals.
gs = 100.0
V = np.array([[0, 0, 0], [gs, 0, 0], [gs, gs, 0], [0, gs, 0]], float)
T = np.array([[0, 1, 2], [0, 2, 3]])
Nn = np.tile([0, 0, 1.0], (4, 1))
ms = pk.sample_surface(V, T, Nn, t_spacing=10.0, random_phi=False, seed=0)
ns = ms.shape[1]
area = gs * gs
exp_s = area / (10.0 ** 2)
# The CVT keeps ~area/t^2 seeds (all distinct after projection).
assert abs(ns - exp_s) <= 2, (ns, exp_s)
poss = ms[7:10, :] + ms[10:13, :]
assert np.allclose(poss[2], 0, atol=1e-6)       # on the plane
# CVT optimizes *even* density (low spread of nearest-neighbor distances)
# rather than a hard minimum separation, so check both.
from scipy.spatial import cKDTree
tree = cKDTree(poss[:2].T)
dd, ii = tree.query(poss[:2].T, k=2)
nn = dd[:, 1]
assert nn.min() >= 3.0, nn.min()
cov = nn.std() / nn.mean()
assert cov < 0.28, cov                          # near-hexagonal evenness
# normal -> +Z
R = ml.rotation_matrix_zxz([ms[16, 0], ms[17, 0], ms[18, 0]])
zs = R @ np.array([0, 0, 1.0])
assert np.allclose(zs, [0, 0, 1], atol=0.05), zs
print("OK surface: %d particles, CoV(NN)=%.3f, normals +Z" % (ns, cov))

# ---- surface jitter: stays on the plane, perturbs off the even layout ------
ms_j = pk.sample_surface(V, T, Nn, t_spacing=10.0, random_phi=False,
                         jitter=3.0, seed=0)
pj = ms_j[7:10, :] + ms_j[10:13, :]
assert np.allclose(pj[2], 0, atol=1e-6)         # tangential jitter -> still z=0
# same seed => same CVT layout, so the difference is the jitter displacement
disp = np.linalg.norm((pj - poss)[:2], axis=0)
assert disp.max() <= 3.0 + 1e-6, disp.max()     # bounded by the jitter radius
assert disp.mean() > 0.5, disp.mean()           # actually moved the picks
print("OK surface jitter: max disp %.2f voxels, still on plane" % disp.max())

# ---- surface on a curved mesh: even, on-surface, no close-pair tail --------
def _uv_sphere(r=50.0, nu=24, nv=48):
    u = np.linspace(0, np.pi, nu)
    az = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, Az = np.meshgrid(u, az, indexing="ij")
    Vv = np.stack([r * np.sin(U) * np.cos(Az), r * np.sin(U) * np.sin(Az),
                   r * np.cos(U)], -1).reshape(-1, 3)
    tr = []
    for i in range(nu - 1):
        for j in range(nv):
            a = i * nv + j; b = i * nv + (j + 1) % nv
            cc = (i + 1) * nv + j; dd = (i + 1) * nv + (j + 1) % nv
            tr += [[a, cc, b], [b, cc, dd]]
    return Vv, np.array(tr), Vv / r

Vsp, Tsp, Nsp = _uv_sphere(50.0)
msp = pk.sample_surface(Vsp, Tsp, Nsp, t_spacing=10.0, random_phi=True, seed=0)
tgt = round(4 * np.pi * 50.0 ** 2 / 100.0)
nsp = msp.shape[1]
assert 0.85 * tgt <= nsp <= tgt, (nsp, tgt)
psp = (msp[7:10] + msp[10:13]).T
rr = np.linalg.norm(psp, axis=1)
assert np.allclose(rr, 50.0, atol=2.0), (rr.min(), rr.max())   # stays on sphere
dsp, _ = cKDTree(psp).query(psp, k=2)
nnsp = dsp[:, 1]; msp_mean = nnsp.mean()
assert nnsp.std() / msp_mean < 0.22, nnsp.std() / msp_mean      # even on curvature
# the repulsion cleanup clears the CVT chord-metric close-pair tail
assert (nnsp < 0.6 * msp_mean).sum() <= 1, int((nnsp < 0.6 * msp_mean).sum())
print("OK surface (sphere): %d particles, CoV=%.3f, no close-pair tail"
      % (nsp, nnsp.std() / msp_mean))

# ---- surface component_number -> _object + by-component half-sets ----------
def _grid_plane(w=120.0, h=40.0, nx=24, ny=8):
    xs = np.linspace(0, w, nx + 1); ys = np.linspace(0, h, ny + 1)
    vid = {}; verts = []
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            vid[(i, j)] = len(verts); verts.append([xx, yy, 0.0])
    tris = []; comp = []
    for j in range(ny):
        for i in range(nx):
            a = vid[(i, j)]; b = vid[(i + 1, j)]
            cc = vid[(i + 1, j + 1)]; d = vid[(i, j + 1)]
            tris += [[a, b, cc], [a, cc, d]]
            lab = 0 if 0.5 * (xs[i] + xs[i + 1]) < w / 2 else 1   # left=0, right=1
            comp += [lab, lab]
    return (np.array(verts), np.array(tris),
            np.tile([0, 0, 1.0], (len(verts), 1)), np.array(comp))

Vg, Tg, Ng, Cg = _grid_plane()
mg = pk.sample_surface(Vg, Tg, Ng, t_spacing=6.0, random_phi=False,
                       component=Cg, seed=0)
pg = (mg[7:10] + mg[10:13]).T
og = mg[5].astype(int)
# each particle's _object is the component of the region it fell in
assert set(og[pg[:, 0] < 58]) == {0}, set(og[pg[:, 0] < 58])
assert set(og[pg[:, 0] > 62]) == {1}, set(og[pg[:, 0] > 62])
# STOPGAP export by component: every object lands wholly in one half, both used
ml.write_stopgap_star("/tmp/comp.star", mg, halfset_by_object=True)
crows = [l for l in open("/tmp/comp.star").read().splitlines()
         if l and not l.startswith(("_", "data_")) and l != "loop_"]
oc = list(ml._STOPGAP_COLUMNS).index("_object")
hc = list(ml._STOPGAP_COLUMNS).index("_halfset")
by_obj = {}
for l in crows:
    f = l.split("\t"); by_obj.setdefault(f[oc], set()).add(f[hc])
assert all(len(h) == 1 for h in by_obj.values()), by_obj   # component not split
assert set().union(*by_obj.values()) == {"A", "B"}, by_obj  # both halves used
# without the flag, the same motl alternates instead
ml.write_stopgap_star("/tmp/comp_alt.star", mg)
arows = [l for l in open("/tmp/comp_alt.star").read().splitlines()
         if l and not l.startswith(("_", "data_")) and l != "loop_"]
alt = [l.split("\t")[hc] for l in arows]
assert alt[:2] == ["A", "B"], alt[:2]
print("OK surface components: _object=component, half-sets split by component")

# ---- multiple spheres -> half-sets split by sphere --------------------------
ms3 = pk.sample_sphere(np.array([[0., 0, 0], [300, 0, 0], [0, 300, 0]]), 30.0,
                       12.0, random_phi=False, tomo_id=0)
assert pk.has_multiple_objects(ms3)          # 3 spheres -> >= 2 objects
assert not pk.has_multiple_objects(m)        # single sphere (top of file) -> no
ml.write_stopgap_star("/tmp/sph3.star", ms3, halfset_by_object=True)
srows = [l for l in open("/tmp/sph3.star").read().splitlines()
         if l and not l.startswith(("_", "data_")) and l != "loop_"]
soc = list(ml._STOPGAP_COLUMNS).index("_object")
shc = list(ml._STOPGAP_COLUMNS).index("_halfset")
byo = {}
for l in srows:
    f = l.split("\t"); byo.setdefault(f[soc], set()).add(f[shc])
assert len(byo) == 3, byo                     # one _object per sphere
assert all(len(h) == 1 for h in byo.values()), byo   # each sphere in one half
assert set().union(*byo.values()) == {"A", "B"}, byo  # both halves used
print("OK multiple spheres: half-sets split by sphere (%d objects)" % len(byo))

# ---- combine / shift / star -------------------------------------------
c = ml.combine_motls([mf, mt], renumber=True)
assert c.shape[1] == nf + nt
assert (c[3, :] == np.arange(1, nf + nt + 1)).all()
sh = ml.shift_motls(mf, shift=(5, 0, 0), apix_in=1.0, apix_out=1.0)
assert sh.shape == mf.shape
ml.write_stopgap_star("/tmp/test_pp.star", mf)
with open("/tmp/test_pp.star") as f:
    lines = f.read().splitlines()
# STOPGAP-compatible format: bare tags (no "#N"), blank line before data.
tag_lines = [l for l in lines if l.startswith("_")]
assert tag_lines and all("#" not in l for l in tag_lines), tag_lines
last_tag = max(i for i, l in enumerate(lines) if l.startswith("_"))
assert lines[last_tag + 1].strip() == "", "need a blank line after the tags"
body = [l for l in lines if l and not l.startswith(("_", "data_")) and l != "loop_"]
assert len(body) == nf, (len(body), nf)   # no particle dropped
# halfset is STOPGAP's string field: values must be 'A'/'B', not 1/2, and both
# halves must appear so a gold-standard split is possible.
hs_col = list(ml._STOPGAP_COLUMNS).index("_halfset")
halves = {row.split("\t")[hs_col] for row in body}
assert halves == {"A", "B"}, halves
print("OK combine/shift/star (STOPGAP tags bare, blank line, halfset A/B)")

print("ALL PICKING TESTS PASSED")
