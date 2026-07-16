"""Headless export tests. Run: ChimeraX --nogui --exit --script test_export.py

Cross-validates Geopickr's Dynamo/RELION angle conventions against the installed
ArtiaX bundle (the authoritative reference) when available.
"""
import numpy as np
from chimerax.geopickr import motivelist as ml
from chimerax.geopickr import export as ex
from chimerax.geopickr import picking as pk

session = session  # provided by ChimeraX script runner
np.random.seed(0)


def rand_motl(n):
    m = np.zeros((20, n))
    m[16:19, :] = np.random.uniform(-180, 180, (3, n))   # phi, psi, theta
    m[7:10, :] = np.random.uniform(0, 200, (3, n))
    return m


# ---- Dynamo matrix<->euler round trip --------------------------------------
for _ in range(50):
    a = np.random.uniform(-180, 180, 3)
    R = ml.rotation_matrix_zxz([a[0], a[1], a[2]])
    td, ti, na = ml.dynamo_matrix2euler(R)
    assert np.allclose(R, ml.dynamo_euler2matrix(td, ti, na), atol=1e-6)
print("OK dynamo matrix<->euler round-trip")

# ---- cross-validate against ArtiaX -----------------------------------------
artiax_ok = False
try:
    from chimerax.artiax.io.Dynamo.DynamoParticleData import DynamoEulerRotation
    from chimerax.artiax.io.RELION.RELIONParticleData import RELIONEulerRotation
    artiax_ok = True
except Exception as e:
    print("NOTE: ArtiaX not importable (%s) — skipping cross-check" % e)

motl = rand_motl(40)
R_tom = ml.tom_rotation_matrices(motl)            # (N,3,3) object orientations
dyn = ml.tom_to_dynamo_eulers(motl)
rel = ml.tom_to_relion_eulers(motl)

if artiax_ok:
    drot = DynamoEulerRotation()
    rrot = RELIONEulerRotation()
    for i in range(motl.shape[1]):
        Rd = np.array(drot.as_place(*dyn[i]).matrix)[:, :3]
        assert np.allclose(Rd, R_tom[i], atol=1e-5), ("dynamo", i, Rd, R_tom[i])
        Rr = np.array(rrot.as_place(*rel[i]).matrix)[:, :3]
        assert np.allclose(Rr, R_tom[i], atol=1e-5), ("relion", i, Rr, R_tom[i])
    print("OK Dynamo + RELION eulers reproduce the orientation via ArtiaX")
else:
    # fallback: our own reconstruction must reproduce R_tom
    for i in range(motl.shape[1]):
        td, ti, na = dyn[i]
        assert np.allclose(ml.dynamo_euler2matrix(td, ti, na), R_tom[i], atol=1e-5)
    print("OK Dynamo eulers reproduce orientation (self-check)")

# ---- writers: parse back ----------------------------------------------------
N = motl.shape[1]
pos1 = np.round(motl[7:10].T) + 1
shifts = np.zeros((N, 3))
ml.write_dynamo_tbl("/tmp/ge.tbl", motl, pos1, shifts, tomo_id=4)
rows = [l.split() for l in open("/tmp/ge.tbl") if l.strip()]
assert len(rows) == N and len(rows[0]) == 35
assert int(rows[0][19]) == 4                       # col 20 tomo id
# col 7-9 must equal tom_to_dynamo_eulers
assert np.allclose([float(x) for x in rows[0][6:9]], dyn[0], atol=1e-3), rows[0][6:9]
print("OK dynamo .tbl writer (35 cols, direct euler, tomo id)")

ml.write_relion5_star("/tmp/ge5.star", motl, motl[7:10].T, tomo_name="TS01")
txt5 = open("/tmp/ge5.star").read()
assert "data_particles" in txt5 and "_rlnCenteredCoordinateXAngst" in txt5
assert "_rlnTomoSubtomogramRot" in txt5
print("OK relion5 star writer")

ml.write_relion3_star("/tmp/ge3.star", motl, motl[7:10].T, tomo_name="TS01")
assert "_rlnCoordinateX" in open("/tmp/ge3.star").read()
print("OK relion3 star writer")

# ---- .vll append ------------------------------------------------------------
with open("/tmp/ge.vll", "w") as f:
    f.write("/data/tomo1.mrc\n/data/tomo2.mrc\n")
assert ml.append_vll_table_ref("/tmp/ge.vll", "/tmp/ge.tbl", tomo_substr="tomo2") == 2
assert "> /tmp/ge.tbl" in open("/tmp/ge.vll").read()
print("OK .vll append")

# ---- coordinate conversion through a real Volume ---------------------------
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data
grid = ArrayGridData(np.zeros((100, 100, 100), np.float32), origin=(0, 0, 0), step=(10, 10, 10))
vol = volume_from_grid_data(grid, session)

class _Stub:
    source_path = ""; name = "test"; deleted = False
m1 = np.zeros((20, 1)); m1[7:10, 0] = [50, 60, 70]    # scene Å -> ijk (5,6,7)
stub = _Stub(); stub.motl = m1

ex.export_model(session, stub, "/tmp/gc.tbl", "dynamo_tbl", volume=vol, tomo_id=1)
r = open("/tmp/gc.tbl").read().split()
assert [int(float(x)) for x in r[23:26]] == [6, 7, 8], r[23:26]    # round(ijk)+1
ex.export_model(session, stub, "/tmp/gc5.star", "relion5", volume=vol, tomo_name="T")
last = [l for l in open("/tmp/gc5.star").read().splitlines()
        if l and not l.startswith(("_", "data_", "loop_", "#"))][-1].split()
cx, cy, cz = float(last[3]), float(last[4]), float(last[5])
assert np.allclose([cx, cy, cz], [(5-50)*10, (6-50)*10, (7-50)*10]), (cx, cy, cz)  # center=size/2
print("OK Volume coord conversion (Dynamo 1-indexed, RELION5 centered Å @ size/2)")

# ---- pick-time offset (baked along +Z) -------------------------------------
c0 = np.array([[0.0, 0.0, 0.0]])
m_no = pk.sample_sphere(c0, 50.0, 10.0, random_phi=False)
m_off = pk.sample_sphere(c0, 50.0, 10.0, random_phi=False, offset=10.0)
r_no = np.linalg.norm(m_no[7:10] + m_no[10:13], axis=0)
r_off = np.linalg.norm(m_off[7:10] + m_off[10:13], axis=0)
assert np.allclose(r_no, 50.0, atol=1.0), r_no.mean()
assert np.allclose(r_off, 60.0, atol=1.0), r_off.mean()      # +Z = outward
m_neg = pk.sample_sphere(c0, 50.0, 10.0, random_phi=False, offset=-10.0)
r_neg = np.linalg.norm(m_neg[7:10] + m_neg[10:13], axis=0)
assert np.allclose(r_neg, 40.0, atol=1.0), r_neg.mean()      # negative = inward
print("OK pick-time offset moves particles along +Z (%.0f -> %.0f / %.0f)"
      % (r_no.mean(), r_off.mean(), r_neg.mean()))

# ---- bake_offsets (display offset -> coordinates) --------------------------
mm = np.zeros((20, 1))
mm[7:10, 0] = [100, 200, 300]
mm[17, 0], mm[18, 0] = ml.normal_to_zxz([0, 0, 1.0])         # +Z along world +Z
baked = ml.bake_offsets(mm, z_offset=7.0)
pos = baked[7:10, 0] + baked[10:13, 0]
assert np.allclose(pos, [100, 200, 307], atol=1e-6), pos     # moved +7 along Z
assert np.allclose(ml.bake_offsets(mm, phi_offset=30)[16, 0], 30.0)
print("OK bake_offsets folds Z/phi display offset into coordinates")

print("ALL EXPORT TESTS PASSED")
