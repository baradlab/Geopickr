# vim: set expandtab ts=4 sw=4:
"""The PlacedParticles model: one instanced surface for a whole motive list.

Rather than creating one Model per particle (as the original Chimera plugin
did), a single :class:`chimerax.core.models.Surface` is used with one geometry
copy positioned by a ``Places`` array.  Per-particle colour is set through the
drawing's per-instance ``colors`` array and per-particle visibility through
``display_positions``.  This scales to tens of thousands of particles.
"""

import numpy as np
from chimerax.core.models import Surface

from . import motivelist as ml
from . import shapes

# Class colour palette (RGBA 0-255), ported from the original plugin's pastel
# set; indexed by (class number mod 10).
_CLASS_PALETTE = np.array([
    (179, 179, 179, 255), (255, 255, 179, 255), (179, 255, 255, 255),
    (179, 179, 255, 255), (255, 179, 255, 255), (255, 179, 179, 255),
    (179, 255, 179, 255), (230, 191, 153, 255), (153, 191, 230, 255),
    (204, 204, 153, 255),
], dtype=np.uint8)

# Visualization modes for display filtering.
SHOW_ALL = "all"
SHOW_NONE = "none"
SHOW_ONE = "one"
SHOW_CC = "cc"
SHOW_CLASS = "class"


class PlacedParticles(Surface):
    """A surface model placing one object per particle of a motive list."""

    SESSION_SAVE = True

    def __init__(self, session, name, motl, source_path="",
                 shape_name="Sphere", custom_path=""):
        super().__init__(name, session)

        self.source_path = source_path
        # Keep the raw matrix so saved motive lists preserve the original
        # (optional) rows; self.motl has rows 13-15 set to coord + shift.
        self.raw_motl = np.array(motl, dtype=np.float64, copy=True)
        self.motl = ml.prepare_motivelist(motl)

        # Object shape
        self.shape_name = shape_name
        self.custom_path = custom_path

        # Placement parameters
        self.voxel_size = 1.0
        self.z_offset = 0.0
        self.phi_offset = 0.0

        # Colour parameters
        self.color_mode = "class"          # "class" | "cc" | "solid"
        self.class_row = ml.ROW_CLASS + 1  # 1-based row used for class colour
        cc = self.motl[ml.ROW_CCC, :]
        self.cc_min = float(cc.min())
        self.cc_max = float(cc.max())
        self.lower_cc_color = (255, 0, 0, 255)
        self.upper_cc_color = (0, 255, 0, 255)
        self.solid_color_rgba = (128, 128, 128, 255)

        # Display parameters
        self.show_mode = SHOW_ALL
        self.show_one_index = 1            # 1-based particle index
        self.show_cc_low = self.cc_min
        self.show_cc_high = self.cc_max
        self.show_class = int(self.class_values().min())

        self._build()

    # -- introspection -------------------------------------------------------
    @property
    def num_particles(self):
        return self.motl.shape[1]

    @property
    def data_cc_range(self):
        cc = self.motl[ml.ROW_CCC, :]
        return float(cc.min()), float(cc.max())

    def class_values(self):
        return self.motl[self.class_row - 1, :]

    def class_range(self):
        cv = self.class_values()
        return int(cv.min()), int(cv.max())

    # -- geometry ------------------------------------------------------------
    def _shape_path(self):
        if self.shape_name == shapes.CUSTOM_SHAPE:
            return self.custom_path
        return shapes.builtin_path(self.shape_name)

    def _build(self):
        """(Re)load the object geometry and rebuild placements and colours."""
        path = self._shape_path()
        if not path:
            return
        va, na, ta = shapes.load_geometry(path)
        self.set_geometry(va, na, ta)
        self.update_placements()
        self.update_colors()
        self.update_display()

    def set_shape(self, shape_name, custom_path=None):
        self.shape_name = shape_name
        if custom_path is not None:
            self.custom_path = custom_path
        self._build()

    def update_placements(self):
        from chimerax.geometry import Places
        pa = ml.placement_array(self.motl, self.voxel_size,
                                self.z_offset, self.phi_offset)
        self.positions = Places(place_array=pa)
        # Re-apply colours/visibility since the position count is unchanged but
        # the Places object was replaced.
        self.update_colors()
        self.update_display()

    # -- colours -------------------------------------------------------------
    def compute_colors(self):
        n = self.num_particles
        if self.color_mode == "solid":
            colors = np.empty((n, 4), np.uint8)
            colors[:] = self.solid_color_rgba
            return colors
        if self.color_mode == "cc":
            cc = self.motl[ml.ROW_CCC, :]
            lo, hi = self.cc_min, self.cc_max
            if hi > lo:
                t = np.clip((cc - lo) / (hi - lo), 0.0, 1.0)
            else:
                t = np.zeros(n)
            c0 = np.array(self.lower_cc_color, dtype=np.float64)
            c1 = np.array(self.upper_cc_color, dtype=np.float64)
            colors = (c0 * (1.0 - t)[:, None] + c1 * t[:, None])
            return np.clip(colors, 0, 255).astype(np.uint8)
        # default: class colouring
        idx = np.mod(self.class_values().astype(np.int64), len(_CLASS_PALETTE))
        return _CLASS_PALETTE[idx]

    def update_colors(self):
        self.colors = self.compute_colors()

    # -- display filtering ---------------------------------------------------
    def display_mask(self):
        n = self.num_particles
        mode = self.show_mode
        if mode == SHOW_NONE:
            return np.zeros(n, bool)
        if mode == SHOW_ONE:
            mask = np.zeros(n, bool)
            i = self.show_one_index - 1
            if 0 <= i < n:
                mask[i] = True
            return mask
        if mode == SHOW_CC:
            cc = self.motl[ml.ROW_CCC, :]
            return (cc >= self.show_cc_low) & (cc <= self.show_cc_high)
        if mode == SHOW_CLASS:
            return self.class_values() == self.show_class
        return np.ones(n, bool)  # SHOW_ALL

    def update_display(self):
        self.display_positions = self.display_mask()

    # -- actions -------------------------------------------------------------
    def displayed_indices(self):
        return np.where(self.display_mask())[0]

    def save_displayed(self, path, fmt=None):
        """Write the currently displayed particles to a new motive list.

        Format is chosen by ``fmt`` ('em' or 'star') or inferred from the file
        extension (default 'em').  The original (pre-position-computation)
        columns are preserved.
        """
        idx = self.displayed_indices()
        out = np.array(self.raw_motl[:, idx], copy=True)
        if fmt is None:
            fmt = "star" if path.lower().endswith(".star") else "em"
        if fmt == "star":
            ml.write_stopgap_star(path, out)
        else:
            ml.write_em_motivelist(path, out)
        return len(idx)

    # -- session save/restore ------------------------------------------------
    def take_snapshot(self, session, flags):
        return {
            "version": 1,
            "name": self.name,
            "source_path": self.source_path,
            "motl": self.raw_motl,
            "shape_name": self.shape_name,
            "custom_path": self.custom_path,
            "voxel_size": self.voxel_size,
            "z_offset": self.z_offset,
            "phi_offset": self.phi_offset,
            "color_mode": self.color_mode,
            "class_row": self.class_row,
            "cc_min": self.cc_min,
            "cc_max": self.cc_max,
            "lower_cc_color": self.lower_cc_color,
            "upper_cc_color": self.upper_cc_color,
            "solid_color_rgba": self.solid_color_rgba,
            "show_mode": self.show_mode,
            "show_one_index": self.show_one_index,
            "show_cc_low": self.show_cc_low,
            "show_cc_high": self.show_cc_high,
            "show_class": self.show_class,
            "model state": Surface.take_snapshot(self, session, flags),
        }

    @classmethod
    def restore_snapshot(cls, session, data):
        m = cls(session, data["name"], data["motl"],
                source_path=data.get("source_path", ""),
                shape_name=data.get("shape_name", "Sphere"),
                custom_path=data.get("custom_path", ""))
        for key in ("voxel_size", "z_offset", "phi_offset", "color_mode",
                    "class_row", "cc_min", "cc_max", "lower_cc_color",
                    "upper_cc_color", "solid_color_rgba", "show_mode",
                    "show_one_index", "show_cc_low", "show_cc_high",
                    "show_class"):
            if key in data:
                setattr(m, key, data[key])
        ms = data.get("model state")
        if ms is not None:
            Surface.set_state_from_snapshot(m, session, ms)
        m._build()
        return m
