# vim: set expandtab ts=4 sw=4:
"""Geometry Picker panel (Qt) - one tab of the Geopickr window."""

import os
from functools import partial

import numpy as np

from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, Surface
from chimerax.core.commands import run

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QListWidget, QListWidgetItem, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog, QLineEdit,
    QScrollArea,
)

from . import picking
from . import motivelist as ml
from .objmodel import PlacedParticles
from .widgets import FloatSlider

STYLES = ("Sphere", "Tube", "Filament", "Surface")


class GeometryPickerPanel:
    """Geometrically pick particles from markers/surfaces. Lives inside a tab."""

    def __init__(self, tool):
        self.tool = tool
        self.session = tool.session
        self.tool_window = tool.tool_window     # shared, used as dialog parent
        self._preview_models = []
        self._last_motl = None
        self._last_model = None
        self._last_name = "picked"

        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.widget.setLayout(layout)

        self._build_markers(layout)
        self._build_style(layout)
        self._build_geometry(layout)
        self._build_sampling(layout)
        self._build_actions(layout)
        self._build_utilities(layout)

        self.status = QLabel("Pick particles from markers or a surface.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._handlers = [
            self.session.triggers.add_handler(ADD_MODELS, lambda *a: self._refresh_sources()),
            self.session.triggers.add_handler(REMOVE_MODELS, lambda *a: self._refresh_sources()),
        ]
        self._refresh_sources()
        self._update_enabled()

    # ------------------------------------------------------------------ UI
    def _build_markers(self, layout):
        box = QGroupBox("Markers (sphere / tube / filament)")
        v = QVBoxLayout(box)
        self.marker_list = QListWidget()
        self.marker_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.marker_list.setMaximumHeight(90)
        v.addWidget(self.marker_list)
        row = QHBoxLayout()
        b_open = QPushButton("Open .cmm...")
        b_open.clicked.connect(self._open_cmm)
        b_new = QPushButton("New marker set")
        b_new.clicked.connect(self._new_marker_set)
        row.addWidget(b_open)
        row.addWidget(b_new)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(box)

    def _build_style(self, layout):
        box = QGroupBox("Object style")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.style_group = QButtonGroup(box)
        self.style_buttons = {}
        for s in STYLES:
            rb = QRadioButton(s)
            self.style_group.addButton(rb)
            self.style_buttons[s] = rb
            row.addWidget(rb)
        # Set the default before connecting, so the handler (which touches
        # widgets built later) doesn't fire during construction.
        self.style_buttons["Sphere"].setChecked(True)
        for rb in self.style_buttons.values():
            rb.toggled.connect(self._style_changed)
        v.addLayout(row)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Surface mesh:"))
        self.surface_combo = QComboBox()
        b_mesh = QPushButton("Open mesh...")
        b_mesh.clicked.connect(self._open_mesh)
        srow.addWidget(self.surface_combo, 1)
        srow.addWidget(b_mesh)
        v.addLayout(srow)
        layout.addWidget(box)

    def _build_geometry(self, layout):
        box = QGroupBox("Geometry (radius fitting)")
        v = QVBoxLayout(box)
        mrow = QHBoxLayout()
        self.perobject_check = QCheckBox("Per-object radii")
        self.perobject_check.toggled.connect(self._perobject_toggled)
        b_fit = QPushButton("Show fit")
        b_fit.clicked.connect(self._show_fit)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear_preview)
        mrow.addWidget(self.perobject_check)
        mrow.addSpacing(12)
        mrow.addWidget(b_fit)
        mrow.addWidget(b_clear)
        mrow.addStretch(1)
        v.addLayout(mrow)

        # Single radius for the whole model (default mode).
        grow = QHBoxLayout()
        lbl = QLabel("Radius")
        lbl.setMinimumWidth(90)
        self.radius_slider = FloatSlider(decimals=1)
        self.radius_slider.setRange(1.0, 300.0)
        self.radius_slider.setValue(20.0)
        self.radius_slider.valueChanged.connect(self._global_radius_changed)
        grow.addWidget(lbl)
        grow.addWidget(self.radius_slider, 1)
        self.global_row = QWidget()
        self.global_row.setLayout(grow)
        v.addWidget(self.global_row)

        # Per-object radius sliders (one per sphere / tube), shown on demand.
        self.radius_area = QScrollArea()
        self.radius_area.setWidgetResizable(True)
        self.radius_area.setMaximumHeight(150)
        holder = QWidget()
        self.radius_form = QVBoxLayout(holder)
        self.radius_form.setContentsMargins(2, 2, 2, 2)
        self.radius_form.addStretch(1)
        self.radius_area.setWidget(holder)
        self.radius_area.setVisible(False)
        v.addWidget(self.radius_area)
        layout.addWidget(box)
        self._fit_objects = []     # dicts: kind, center/axis/markerset, set_id, label, slider, model

    def _build_sampling(self, layout):
        box = QGroupBox("Sampling (voxels)")
        grid = QGridLayout(box)
        self.tan_spin = self._dspin(0.0, 100000.0, 2, 10.0)
        grid.addWidget(QLabel("Tangential"), 0, 0)
        grid.addWidget(self.tan_spin, 0, 1)
        self.ax_spin = self._dspin(0.0, 100000.0, 2, 0.0)
        grid.addWidget(QLabel("Axial"), 0, 2)
        grid.addWidget(self.ax_spin, 0, 3)
        self.twist_spin = self._dspin(-360.0, 360.0, 2, 0.0)
        grid.addWidget(QLabel("Twist °/step"), 1, 0)
        grid.addWidget(self.twist_spin, 1, 1)
        # Offset moves particles along their +Z (normal) into the coordinates.
        self.offset_spin = self._dspin(-100000.0, 100000.0, 2, 0.0)
        self.offset_spin.valueChanged.connect(self._offset_changed)
        self.offset_spin.setToolTip(
            "Move picked particles along their +Z axis (the surface normal for\n"
            "sphere/tube/surface; the axis tangent for filament). Positive =\n"
            "outward. Baked into the coordinates. The cyan shell in 'Show fit'\n"
            "previews where sphere/tube particles will land.")
        grid.addWidget(QLabel("Offset"), 1, 2)
        grid.addWidget(self.offset_spin, 1, 3)
        self.tomoid_spin = QSpinBox()
        self.tomoid_spin.setRange(0, 1000000)
        grid.addWidget(QLabel("Tomo ID"), 2, 0)
        grid.addWidget(self.tomoid_spin, 2, 1)
        # Jitter randomly perturbs surface picks in-plane to break the regular
        # CVT lattice (surface style only).
        self.jitter_spin = self._dspin(0.0, 100000.0, 2, 0.0)
        self.jitter_spin.setToolTip(
            "Surface only: after the even (CVT) layout, randomly perturb each\n"
            "particle within this radius (voxels) in the surface plane, to break\n"
            "up the regular lattice. 0 = keep the even layout.")
        grid.addWidget(QLabel("Jitter"), 2, 2)
        grid.addWidget(self.jitter_spin, 2, 3)
        self.randphi_check = QCheckBox("Randomize phi")
        self.randphi_check.setChecked(True)
        grid.addWidget(self.randphi_check, 3, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addWidget(box)

    def _build_actions(self, layout):
        row = QHBoxLayout()
        b_pick = QPushButton("Pick")
        b_pick.clicked.connect(self._pick)
        b_save = QPushButton("Export...")
        b_save.clicked.connect(self._save)
        row.addWidget(b_pick)
        row.addWidget(b_save)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_utilities(self, layout):
        box = QGroupBox("Utilities (combine / shift)")
        box.setCheckable(True)
        box.setChecked(False)
        grid = QGridLayout(box)

        # Combine: paired input/output line edits on one row.
        grid.addWidget(QLabel("<b>Combine</b>"), 0, 0, 1, 4)
        self.comb_in = QLineEdit()
        self.comb_in.setPlaceholderText("input glob, e.g. /data/*_sphere.em")
        self.comb_out = QLineEdit()
        self.comb_out.setPlaceholderText("output .em")
        grid.addWidget(self.comb_in, 1, 0, 1, 2)
        grid.addWidget(self.comb_out, 1, 2, 1, 2)
        crow = QHBoxLayout()
        self.comb_renum = QCheckBox("Renumber")
        b_comb = QPushButton("Combine")
        b_comb.clicked.connect(self._combine)
        crow.addWidget(self.comb_renum)
        crow.addStretch(1)
        crow.addWidget(b_comb)
        grid.addLayout(crow, 2, 0, 1, 4)

        # Shift: paired input/output, then offset and rescale rows.
        grid.addWidget(QLabel("<b>Shift / rescale</b>"), 3, 0, 1, 4)
        self.shift_in = QLineEdit()
        self.shift_in.setPlaceholderText("input .em")
        self.shift_out = QLineEdit()
        self.shift_out.setPlaceholderText("output .em")
        grid.addWidget(self.shift_in, 4, 0, 1, 2)
        grid.addWidget(self.shift_out, 4, 2, 1, 2)
        self.sx = self._dspin(-1e6, 1e6, 2, 0.0)
        self.sy = self._dspin(-1e6, 1e6, 2, 0.0)
        self.sz = self._dspin(-1e6, 1e6, 2, 0.0)
        off = QHBoxLayout()
        off.addWidget(QLabel("Offset"))
        for lab, w in (("X", self.sx), ("Y", self.sy), ("Z", self.sz)):
            off.addWidget(QLabel(lab))
            off.addWidget(w)
        off.addStretch(1)
        grid.addLayout(off, 5, 0, 1, 4)
        self.apin = self._dspin(1e-6, 1e6, 3, 1.0)
        self.apout = self._dspin(1e-6, 1e6, 3, 1.0)
        ap = QHBoxLayout()
        ap.addWidget(QLabel("Rescale"))
        ap.addWidget(QLabel("in"))
        ap.addWidget(self.apin)
        ap.addWidget(QLabel("out"))
        ap.addWidget(self.apout)
        ap.addStretch(1)
        b_shift = QPushButton("Shift")
        b_shift.clicked.connect(self._shift)
        ap.addWidget(b_shift)
        grid.addLayout(ap, 6, 0, 1, 4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)
        layout.addWidget(box)

    @staticmethod
    def _dspin(lo, hi, dec, val):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setKeyboardTracking(False)
        s.setValue(val)
        s.setMaximumWidth(72)
        return s

    # -------------------------------------------------------------- sources
    def _marker_sets(self):
        from chimerax.markers import MarkerSet
        return [m for m in self.session.models.list(type=MarkerSet)]

    def _surfaces(self):
        out = []
        from chimerax.markers import MarkerSet
        for m in self.session.models.list():
            if isinstance(m, Surface) and not isinstance(m, (PlacedParticles, MarkerSet)):
                if getattr(m, "triangles", None) is not None and len(m.triangles):
                    out.append(m)
        return out

    def _refresh_sources(self):
        msets = self._marker_sets()
        self.marker_list.blockSignals(True)
        sel = {i.text() for i in self.marker_list.selectedItems()}
        self.marker_list.clear()
        for m in msets:
            it = QListWidgetItem("#%s  %s  (%d markers)"
                                 % (m.id_string, m.name, m.num_atoms))
            self.marker_list.addItem(it)
            if it.text() in sel:
                it.setSelected(True)
        self.marker_list.blockSignals(False)
        self._mset_models = msets

        surfs = self._surfaces()
        cur = self.surface_combo.currentText()
        self.surface_combo.blockSignals(True)
        self.surface_combo.clear()
        for m in surfs:
            self.surface_combo.addItem("#%s  %s" % (m.id_string, m.name))
        self._surf_models = surfs
        i = self.surface_combo.findText(cur)
        if i >= 0:
            self.surface_combo.setCurrentIndex(i)
        self.surface_combo.blockSignals(False)

    def _selected_marker_models(self):
        rows = [self.marker_list.row(i) for i in self.marker_list.selectedItems()]
        if not rows:                       # default: use all
            return list(self._mset_models)
        return [self._mset_models[r] for r in sorted(rows)]

    def _current_surface(self):
        i = self.surface_combo.currentIndex()
        if 0 <= i < len(self._surf_models):
            return self._surf_models[i]
        return None

    def _style(self):
        for s, rb in self.style_buttons.items():
            if rb.isChecked():
                return s
        return "Sphere"

    def _style_changed(self, checked):
        if checked:
            if getattr(self, "_fit_objects", None):
                self._clear_fit()
            self._update_enabled()

    def _update_enabled(self):
        s = self._style()
        radius_ok = s in ("Sphere", "Tube")
        self.radius_slider.setEnabled(radius_ok)
        self.perobject_check.setEnabled(radius_ok)
        self.tan_spin.setEnabled(s in ("Sphere", "Tube", "Surface"))
        self.ax_spin.setEnabled(s in ("Tube", "Filament"))
        self.twist_spin.setEnabled(s == "Filament")
        self.jitter_spin.setEnabled(s == "Surface")
        self.surface_combo.setEnabled(s == "Surface")
        self.marker_list.setEnabled(s != "Surface")

    # --------------------------------------------------------------- actions
    def _new_marker_set(self):
        from .place_points_tool import new_marker_set
        new_marker_set(self.session)
        self._refresh_sources()

    def _open_cmm(self):
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area, "Open marker .cmm", "",
            "Chimera markers (*.cmm);;All files (*)")
        if path:
            run(self.session, 'open "%s"' % path)
            self._refresh_sources()

    def _open_mesh(self):
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area, "Open surface mesh", "",
            "Surfaces (*.obj *.ply *.stl);;All files (*)")
        if path:
            run(self.session, 'open "%s"' % path)
            self._refresh_sources()

    # ----------------------------------------------------- radius fit / preview
    def _object_radius(self, obj):
        """Radius for one fit object: its own slider, or the global slider."""
        sl = obj.get("slider")
        return sl.value() if sl is not None else self.radius_slider.value()

    def _clear_fit(self):
        for obj in self._fit_objects:
            for key in ("model", "offset_model"):
                m = obj.get(key)
                if m is not None and not m.deleted:
                    self.session.models.close([m])
            w = obj.get("widget")
            if w is not None:
                w.setParent(None)
        self._fit_objects = []

    def _clear_preview(self):
        self._clear_fit()
        self.status.setText("Cleared fit preview.")

    def _perobject_toggled(self, checked):
        self.global_row.setVisible(not checked)
        self.radius_area.setVisible(checked)
        if self._fit_objects:          # rebuild in the new mode
            self._show_fit()

    def _fit_descriptors(self):
        """Return (style, [object descriptors]) for the current selection."""
        s = self._style()
        if s not in ("Sphere", "Tube"):
            return s, []
        objs = []
        for sid, ms in enumerate(self._selected_marker_models()):
            if s == "Sphere":
                for j, c in enumerate(ms.atoms.coords):
                    objs.append({"kind": "sphere", "set_id": sid,
                                 "center": np.array(c, dtype=float),
                                 "label": "%s m%d" % (ms.name, j + 1)})
            elif ms.num_atoms >= 2:
                objs.append({"kind": "tube", "set_id": sid, "markerset": ms,
                             "axis": np.array(ms.atoms.coords, dtype=float),
                             "label": "%s tube" % ms.name})
        return s, objs

    def _show_fit(self):
        """Draw wireframes; in per-object mode also build one slider each."""
        self._clear_fit()
        s, objs = self._fit_descriptors()
        if s not in ("Sphere", "Tube"):
            self.status.setText("Radius fitting applies to Sphere and Tube.")
            return
        if not objs:
            self.status.setText("Open/select a marker set "
                                "(>=2 markers per tube).")
            return
        per_object = self.perobject_check.isChecked()
        default_r = self.radius_slider.value()
        for obj in objs:
            obj["model"] = None
            if per_object:
                w = QWidget()
                h = QHBoxLayout(w)
                h.setContentsMargins(0, 0, 0, 0)
                lab = QLabel(obj["label"])
                lab.setMinimumWidth(90)
                sl = FloatSlider(decimals=1)
                sl.setRange(1.0, max(300.0, default_r * 4))
                sl.setValue(default_r)
                sl.valueChanged.connect(partial(self._redraw_object, obj))
                h.addWidget(lab)
                h.addWidget(sl, 1)
                self.radius_form.insertWidget(self.radius_form.count() - 1, w)
                obj["widget"] = w
                obj["slider"] = sl
            else:
                obj["slider"] = None
        self._fit_objects = objs
        for obj in objs:
            self._redraw_object(obj)
        self.status.setText(
            "Showing %d object(s) (%s radius). Drag to fit, then Pick."
            % (len(objs), "per-object" if per_object else "single"))

    def _global_radius_changed(self, value):
        for obj in self._fit_objects:
            if obj.get("slider") is None:
                self._redraw_object(obj)

    def _offset_changed(self, _value=None):
        # Update the cyan "particle shell" on every shown fit object.
        for obj in self._fit_objects:
            self._redraw_object(obj)

    def _shell_geometry(self, obj, r):
        if obj["kind"] == "sphere":
            return _sphere_geometry(obj["center"], r)
        return _tube_geometry(obj["axis"], r)

    def _redraw_object(self, obj, _value=None):
        r = self._object_radius(obj)
        va, na, ta = self._shell_geometry(obj, r)
        m = obj.get("model")
        if m is None or m.deleted:
            m = Surface("fit %s" % obj["label"], self.session)
            m.display_style = m.Mesh
            m.color = (255, 255, 0, 180)
            self.session.models.add([m])
            obj["model"] = m
        m.set_geometry(va, na, ta)

        # Cyan shell at r + offset shows where particles will actually land.
        off = self.offset_spin.value()
        shell = obj.get("offset_model")
        if not off:
            if shell is not None and not shell.deleted:
                self.session.models.close([shell])
            obj["offset_model"] = None
            return
        va, na, ta = self._shell_geometry(obj, max(0.5, r + off))
        if shell is None or shell.deleted:
            shell = Surface("offset %s" % obj["label"], self.session)
            shell.display_style = shell.Mesh
            shell.color = (0, 200, 255, 170)
            self.session.models.add([shell])
            obj["offset_model"] = shell
        shell.set_geometry(va, na, ta)

    def _gather_motl(self):
        s = self._style().lower()
        rp = self.randphi_check.isChecked()
        tan = self.tan_spin.value()
        tomo = self.tomoid_spin.value()
        off = self.offset_spin.value()
        if s == "surface":
            surf = self._current_surface()
            if surf is None:
                self.status.setText("Choose a surface mesh first.")
                return None
            return picking.pick(self.session, style="surface",
                                surface_model=surf, tangential=tan,
                                random_phi=rp, tomo_id=tomo, offset=off,
                                jitter=self.jitter_spin.value())

        models = self._selected_marker_models()
        if not models:
            self.status.setText("Open and select a marker set first.")
            return None

        # Use fitted radii when a fit is shown (per-object or single).
        fit = [o for o in self._fit_objects if o["kind"] == s]
        if s == "sphere" and fit:
            centers = np.array([o["center"] for o in fit])
            radii = np.array([self._object_radius(o) for o in fit])
            set_ids = np.array([o["set_id"] for o in fit])
            return picking.sample_sphere(centers, radii, tan, random_phi=rp,
                                         tomo_id=tomo, set_ids=set_ids, offset=off)
        if s == "tube" and fit:
            axis_list = [o["axis"] for o in fit]
            radii = [self._object_radius(o) for o in fit]
            return picking.sample_tube(axis_list, radii, tan,
                                       self.ax_spin.value(), random_phi=rp,
                                       tomo_id=tomo, offset=off)

        return picking.pick(
            self.session, style=s, marker_models=models,
            radius=self.radius_slider.value(), tangential=tan,
            axial=self.ax_spin.value(), twist=self.twist_spin.value(),
            random_phi=rp, tomo_id=tomo, offset=off)

    def _pick(self):
        try:
            motl = self._gather_motl()
        except Exception as e:
            self.status.setText("Pick error: %s" % e)
            self.session.logger.warning("Geometry Picker: %s" % e)
            return
        if motl is None:
            return
        if motl.shape[1] == 0:
            self.status.setText("No particles generated (check spacing/radius).")
            return
        self._clear_preview()
        self._last_motl = motl
        style = self._style()
        if style == "Surface":
            src = self._current_surface()
            base = src.name if src is not None else "surface"
        else:
            sel = self._selected_marker_models()
            base = sel[0].name if sel else "markers"
        self._last_name = "%s (%s)" % (base, style.lower())
        model = PlacedParticles(self.session, self._last_name, motl,
                                shape_name="Hexagon")
        model.voxel_size = 0.1
        model.update_placements()
        self.session.models.add([model])
        self._last_model = model
        self.status.setText("Picked %d particles (%s). See the Place Object tab."
                            % (model.num_particles, style))
        self.tool.show_place_object(model)

    def _save(self):
        if self._last_model is None or self._last_model.deleted:
            self.status.setText("Pick particles before exporting.")
            return
        from . import export
        result = export.run_export_dialog(self.tool, self._last_model)
        if result is not None:
            self.status.setText(result[1])

    # ------------------------------------------------------------ utilities
    def _combine(self):
        from glob import glob
        files = sorted(glob(self.comb_in.text()))
        if len(files) < 2:
            self.status.setText("Combine needs >=2 files matching the glob.")
            return
        if not self.comb_out.text():
            self.status.setText("Specify a combined output file.")
            return
        motls = [ml.read_em_motivelist(f) for f in files]
        out = ml.combine_motls(motls, renumber=self.comb_renum.isChecked())
        ml.write_em_motivelist(self.comb_out.text(), out)
        self.status.setText("Combined %d files (%d particles) -> %s"
                            % (len(files), out.shape[1],
                               os.path.basename(self.comb_out.text())))

    def _shift(self):
        if not os.path.isfile(self.shift_in.text()) or not self.shift_out.text():
            self.status.setText("Specify existing input and an output file.")
            return
        motl = ml.read_em_motivelist(self.shift_in.text())
        out = ml.shift_motls(
            motl, shift=(self.sx.value(), self.sy.value(), self.sz.value()),
            apix_in=self.apin.value(), apix_out=self.apout.value())
        ml.write_em_motivelist(self.shift_out.text(), out)
        self.status.setText("Shifted -> %s" % os.path.basename(self.shift_out.text()))

    # ------------------------------------------------------------ lifecycle
    def close(self):
        for h in self._handlers:
            self.session.triggers.remove_handler(h)
        self._handlers = []
        self._clear_preview()


# ---------------------------------------------------------------------------
# Wireframe geometry for the live radius-fit preview
# ---------------------------------------------------------------------------
def _sphere_geometry(center, radius, nlat=14, nlon=22):
    """UV-sphere (vertices, normals, triangles) centerd at ``center``."""
    lats = np.linspace(0.0, np.pi, nlat + 1)
    lons = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    x = np.sin(la) * np.cos(lo)
    y = np.sin(la) * np.sin(lo)
    z = np.cos(la)
    unit = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    va = (unit * radius + np.asarray(center, float)).astype(np.float32)
    na = unit.astype(np.float32)
    tris = []
    for i in range(nlat):
        for j in range(nlon):
            a = i * nlon + j
            b = i * nlon + (j + 1) % nlon
            c = (i + 1) * nlon + j
            d = (i + 1) * nlon + (j + 1) % nlon
            tris.append((a, b, c))
            tris.append((b, d, c))
    return va, na, np.array(tris, np.int32)


def _tube_geometry(axis_pts, radius, nsides=16):
    """Tube mesh of given radius around a spline through ``axis_pts``."""
    n_ring = max(20, len(axis_pts) * 12)
    pos, tan = ml.natural_cubic_spline(np.asarray(axis_pts, float), n_ring)
    ang = np.linspace(0.0, 2 * np.pi, nsides, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    verts, norms = [], []
    ref = np.array([0.0, 0.0, 1.0])
    for p, t in zip(pos, tan):
        if abs(np.dot(t, ref)) > 0.95:
            ref2 = np.array([1.0, 0.0, 0.0])
        else:
            ref2 = ref
        u = np.cross(t, ref2)
        u /= (np.linalg.norm(u) + 1e-9)
        w = np.cross(t, u)
        ring_n = cos_a[:, None] * u + sin_a[:, None] * w
        verts.append(p + radius * ring_n)
        norms.append(ring_n)
    va = np.concatenate(verts).astype(np.float32)
    na = np.concatenate(norms).astype(np.float32)
    tris = []
    nr = len(pos)
    for i in range(nr - 1):
        for j in range(nsides):
            a = i * nsides + j
            b = i * nsides + (j + 1) % nsides
            c = (i + 1) * nsides + j
            d = (i + 1) * nsides + (j + 1) % nsides
            tris.append((a, b, c))
            tris.append((b, d, c))
    return va, na, np.array(tris, np.int32)
