"""Smoke test run inside ChimeraX nogui: open ChimeraX and run with
   ChimeraX --nogui --exit --script test_smoke.py
"""
import numpy as np
from chimerax.geopickr import motivelist as ml
from chimerax.geopickr.objmodel import PlacedParticles
from chimerax.geopickr import objmodel

session = session  # provided by ChimeraX script runner

# --- build a synthetic 20 x N motive list -------------------------------
N = 5
motl = np.zeros((20, N), dtype=np.float64)
motl[0, :] = np.linspace(0.1, 0.9, N)        # CCC
motl[7, :] = np.arange(N) * 100.0            # X coords
motl[8, :] = 0.0                             # Y
motl[9, :] = 0.0                             # Z
motl[10:13, :] = 1.0                         # shifts
motl[16, :] = [0, 90, 180, 270, 45]          # phi
motl[17, :] = 0.0                            # psi
motl[18, :] = [0, 0, 90, 90, 30]             # theta
motl[19, :] = [1, 2, 3, 1, 2]                # class

path = "/tmp/test_motl.em"
ml.write_em_motivelist(path, motl)
roundtrip = ml.read_em_motivelist(path)
assert roundtrip.shape == (20, N), roundtrip.shape
assert np.allclose(roundtrip, motl), "round-trip mismatch"
print("OK em round-trip", roundtrip.shape)

# --- placement array ----------------------------------------------------
prepared = ml.prepare_motivelist(motl)
assert np.allclose(prepared[13:16, 0], [0, 0, 0] + np.array([1, 1, 1])), prepared[13:16, 0]
pa = ml.placement_array(prepared, voxel_size=2.0)
assert pa.shape == (N, 3, 4), pa.shape
# particle 0: phi=0,psi=0,theta=0 -> identity rotation * voxel; origin = coord+shift
assert np.allclose(pa[0, :, :3], 2.0 * np.eye(3)), pa[0, :, :3]
assert np.allclose(pa[0, :, 3], [1, 1, 1]), pa[0, :, 3]
print("OK placement array")

# --- model --------------------------------------------------------------
m = PlacedParticles(session, "test", motl, source_path=path)
session.models.add([m])
assert m.num_particles == N
assert len(m.positions) == N, len(m.positions)
assert m.colors.shape == (N, 4), m.colors.shape
print("OK model build, colors", m.colors.shape)

# colour modes
m.color_mode = "cc"; m.update_colors()
assert m.colors.shape == (N, 4)
m.color_mode = "solid"; m.solid_color_rgba = (10, 20, 30, 255); m.update_colors()
assert tuple(m.colors[0]) == (10, 20, 30, 255), m.colors[0]
print("OK colour modes")

# display filters
m.show_mode = objmodel.SHOW_NONE; m.update_display()
assert m.display_positions.sum() == 0
m.show_mode = objmodel.SHOW_ONE; m.show_one_index = 3; m.update_display()
assert m.display_positions.sum() == 1 and m.display_positions[2]
m.show_mode = objmodel.SHOW_CLASS; m.class_row = 20; m.show_class = 1; m.update_display()
assert m.display_positions.sum() == 2, m.display_positions.sum()  # classes [1,2,3,1,2]
m.show_mode = objmodel.SHOW_CC; m.show_cc_low = 0.3; m.show_cc_high = 0.7; m.update_display()
exp = ((motl[0] >= 0.3) & (motl[0] <= 0.7)).sum()
assert m.display_positions.sum() == exp, (m.display_positions.sum(), exp)
print("OK display filters")

# shape change to a custom-free builtin
m.set_shape("ArrowZ")
assert m.num_particles == N and len(m.positions) == N
print("OK shape change")

# save displayed
m.show_mode = objmodel.SHOW_ALL; m.update_display()
n = m.save_displayed("/tmp/test_motl_display.em")
assert n == N, n
back = ml.read_em_motivelist("/tmp/test_motl_display.em")
assert back.shape == (20, N)
print("OK save displayed", n)

print("ALL SMOKE TESTS PASSED")
