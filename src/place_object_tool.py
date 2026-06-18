# vim: set expandtab ts=4 sw=4:
"""Place Object panel (Qt) - one tab of the Geopickr window."""

import os

from chimerax.core.models import REMOVE_MODELS

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QSlider,
    QGroupBox, QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
    QFileDialog, QLineEdit, QSizePolicy,
)

from . import shapes
from . import objmodel
from .objmodel import PlacedParticles
from .widgets import FloatSlider, ColorButton


class PlaceObjectPanel:
    """Display motive lists as placed objects. Lives inside a tab."""

    def __init__(self, tool):
        self.tool = tool
        self.session = tool.session
        self.tool_window = tool.tool_window     # shared, used as dialog parent
        self._updating = False                  # guard against feedback loops

        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.widget.setLayout(layout)

        self._build_list_section(layout)
        self._build_object_section(layout)
        self._build_color_section(layout)
        self._build_show_section(layout)
        self._build_action_section(layout)

        self.status = QLabel("Open a motive list (.em) to begin.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        # Refresh the list when models are removed elsewhere.
        self._remove_handler = self.session.triggers.add_handler(
            REMOVE_MODELS, lambda *a: self._refresh_list())

        self.refresh()

    # ===================================================================
    # UI construction
    # ===================================================================
    def _build_list_section(self, layout):
        box = QGroupBox("Motive lists")
        v = QVBoxLayout(box)
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(110)
        self.list_widget.currentRowChanged.connect(lambda i: self.refresh())
        v.addWidget(self.list_widget)
        row = QHBoxLayout()
        open_btn = QPushButton("Open motive list...")
        open_btn.clicked.connect(self._open_motivelist)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._close_selected)
        row.addWidget(open_btn)
        row.addWidget(close_btn)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(box)

    def _build_object_section(self, layout):
        box = QGroupBox("Object")
        form = QFormLayout(box)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(shapes.BUILTIN_SHAPES) + [shapes.CUSTOM_SHAPE])
        self.shape_combo.currentTextChanged.connect(self._shape_changed)
        form.addRow("Shape", self.shape_combo)

        crow = QHBoxLayout()
        self.custom_path_edit = QLineEdit()
        self.custom_path_edit.setReadOnly(True)
        self.custom_browse = QPushButton("Browse...")
        self.custom_browse.clicked.connect(self._browse_custom)
        crow.addWidget(self.custom_path_edit, 1)
        crow.addWidget(self.custom_browse)
        self.custom_widget = QWidget()
        self.custom_widget.setLayout(crow)
        crow.setContentsMargins(0, 0, 0, 0)
        form.addRow("STL file", self.custom_widget)

        self.voxel_spin = self._dspin(0.0001, 100000.0, 4, 1.0)
        self.voxel_spin.valueChanged.connect(self._placement_changed)
        form.addRow("Voxel size", self.voxel_spin)

        self.zoff_spin = self._dspin(-100000.0, 100000.0, 3, 0.0)
        self.zoff_spin.valueChanged.connect(self._placement_changed)
        form.addRow("Z offset", self.zoff_spin)

        self.phi_spin = self._dspin(-360.0, 360.0, 2, 0.0)
        self.phi_spin.valueChanged.connect(self._placement_changed)
        form.addRow("Phi offset (°)", self.phi_spin)

        layout.addWidget(box)

    def _build_color_section(self, layout):
        box = QGroupBox("Colour")
        v = QVBoxLayout(box)

        self.color_group = QButtonGroup(box)
        rb_row = QHBoxLayout()
        self.rb_class = QRadioButton("Class")
        self.rb_cc = QRadioButton("Cross-corr.")
        self.rb_solid = QRadioButton("Solid")
        for rb, mode in ((self.rb_class, "class"), (self.rb_cc, "cc"),
                         (self.rb_solid, "solid")):
            self.color_group.addButton(rb)
            rb.toggled.connect(self._color_mode_changed)
            rb_row.addWidget(rb)
        rb_row.addStretch(1)
        v.addLayout(rb_row)

        grid = QGridLayout()
        # Class row
        grid.addWidget(QLabel("Class row"), 0, 0)
        self.class_row_spin = QSpinBox()
        self.class_row_spin.setRange(1, 20)
        self.class_row_spin.setValue(20)
        self.class_row_spin.valueChanged.connect(self._class_row_changed)
        grid.addWidget(self.class_row_spin, 0, 1)
        # Solid colour
        grid.addWidget(QLabel("Solid colour"), 0, 2)
        self.solid_color_btn = ColorButton((128, 128, 128, 255))
        self.solid_color_btn.colorChanged.connect(self._color_changed)
        grid.addWidget(self.solid_color_btn, 0, 3)
        # CC colour ramp
        grid.addWidget(QLabel("CC range"), 1, 0)
        cc_row = QHBoxLayout()
        self.cc_low_color = ColorButton((255, 0, 0, 255))
        self.cc_low_color.colorChanged.connect(self._color_changed)
        self.cc_min_spin = self._dspin(-1e6, 1e6, 4, 0.0)
        self.cc_min_spin.valueChanged.connect(self._color_changed)
        self.cc_max_spin = self._dspin(-1e6, 1e6, 4, 1.0)
        self.cc_max_spin.valueChanged.connect(self._color_changed)
        self.cc_high_color = ColorButton((0, 255, 0, 255))
        self.cc_high_color.colorChanged.connect(self._color_changed)
        self.cc_auto_btn = QPushButton("Auto")
        self.cc_auto_btn.clicked.connect(self._auto_cc_range)
        cc_row.addWidget(self.cc_low_color)
        cc_row.addWidget(self.cc_min_spin)
        cc_row.addWidget(QLabel("–"))
        cc_row.addWidget(self.cc_max_spin)
        cc_row.addWidget(self.cc_high_color)
        cc_row.addWidget(self.cc_auto_btn)
        cc_w = QWidget()
        cc_w.setLayout(cc_row)
        cc_row.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(cc_w, 1, 1, 1, 3)
        v.addLayout(grid)
        layout.addWidget(box)

    def _build_show_section(self, layout):
        box = QGroupBox("Show")
        v = QVBoxLayout(box)

        self.show_group = QButtonGroup(box)
        rb_row = QHBoxLayout()
        self.show_all = QRadioButton("All")
        self.show_none = QRadioButton("None")
        self.show_one = QRadioButton("Single")
        self.show_cc = QRadioButton("CC range")
        self.show_class = QRadioButton("Class")
        for rb in (self.show_all, self.show_none, self.show_one,
                   self.show_cc, self.show_class):
            self.show_group.addButton(rb)
            rb.toggled.connect(self._show_mode_changed)
            rb_row.addWidget(rb)
        rb_row.addStretch(1)
        v.addLayout(rb_row)

        form = QFormLayout()
        # Single particle index
        prow = QHBoxLayout()
        self.one_slider = QSlider(Qt.Horizontal)
        self.one_slider.setRange(1, 1)
        self.one_spin = QSpinBox()
        self.one_spin.setRange(1, 1)
        self.one_slider.valueChanged.connect(
            lambda val: (self.one_spin.setValue(val), self._show_param_changed()))
        self.one_spin.valueChanged.connect(
            lambda val: (self.one_slider.setValue(val), self._show_param_changed()))
        prow.addWidget(self.one_slider, 1)
        prow.addWidget(self.one_spin)
        self.one_w = QWidget()
        self.one_w.setLayout(prow)
        prow.setContentsMargins(0, 0, 0, 0)
        form.addRow("Particle #", self.one_w)

        # CC threshold range
        self.cc_low_slider = FloatSlider()
        self.cc_low_slider.valueChanged.connect(lambda v: self._show_param_changed())
        form.addRow("CC ≥", self.cc_low_slider)
        self.cc_high_slider = FloatSlider()
        self.cc_high_slider.valueChanged.connect(lambda v: self._show_param_changed())
        form.addRow("CC ≤", self.cc_high_slider)

        # Class number
        self.class_num_spin = QSpinBox()
        self.class_num_spin.setRange(0, 0)
        self.class_num_spin.valueChanged.connect(lambda v: self._show_param_changed())
        form.addRow("Class #", self.class_num_spin)

        v.addLayout(form)
        layout.addWidget(box)

    def _build_action_section(self, layout):
        row = QHBoxLayout()
        focus_btn = QPushButton("Focus")
        focus_btn.clicked.connect(self._focus)
        save_btn = QPushButton("Save shown...")
        save_btn.clicked.connect(self._save_shown)
        row.addWidget(focus_btn)
        row.addWidget(save_btn)
        row.addStretch(1)
        layout.addLayout(row)

    @staticmethod
    def _dspin(lo, hi, decimals, value):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setKeyboardTracking(False)
        s.setValue(value)
        s.setMaximumWidth(80)
        return s

    # ===================================================================
    # Model list management
    # ===================================================================
    def _models(self):
        return [m for m in self.session.models.list(type=PlacedParticles)]

    def current_model(self):
        models = self._models()
        i = self.list_widget.currentRow()
        if 0 <= i < len(models):
            return models[i]
        return None

    def _refresh_list(self):
        models = self._models()
        cur = self.list_widget.currentRow()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for m in models:
            self.list_widget.addItem(QListWidgetItem(
                "#%s  %s  (%d particles)"
                % (m.id_string, m.name, m.num_particles)))
        if models:
            if not (0 <= cur < len(models)):
                cur = len(models) - 1
            self.list_widget.setCurrentRow(cur)
        self.list_widget.blockSignals(False)

    # ===================================================================
    # Loading / closing
    # ===================================================================
    def _open_motivelist(self):
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area, "Open motive list",
            "", "EM motive list (*.em);;All files (*)")
        if not path:
            return
        self.load_motivelist(path)

    def load_motivelist(self, path):
        from . import motivelist as ml
        try:
            motl = ml.read_em_motivelist(path)
        except Exception as e:
            self.status.setText("Error reading %s: %s" % (os.path.basename(path), e))
            self.session.logger.warning("Place Object: %s" % e)
            return None
        name = os.path.basename(path)
        try:
            model = PlacedParticles(self.session, name, motl, source_path=path)
        except Exception as e:
            self.status.setText("Error placing objects: %s" % e)
            self.session.logger.warning("Place Object: %s" % e)
            return None
        self.session.models.add([model])
        self._refresh_list()
        # select the newly added model
        models = self._models()
        if model in models:
            self.list_widget.setCurrentRow(models.index(model))
        self.status.setText(
            "Loaded %s with %d particles." % (name, model.num_particles))
        return model

    def _close_selected(self):
        m = self.current_model()
        if m is not None:
            self.session.models.close([m])
            self._refresh_list()
            self.refresh()

    # ===================================================================
    # Refresh controls from the selected model
    # ===================================================================
    def refresh(self):
        self._refresh_list()
        m = self.current_model()
        enabled = m is not None
        for w in (self.shape_combo, self.voxel_spin, self.zoff_spin,
                  self.phi_spin, self.rb_class, self.rb_cc, self.rb_solid,
                  self.show_all, self.show_none, self.show_one, self.show_cc,
                  self.show_class):
            w.setEnabled(enabled)
        if m is None:
            return

        self._updating = True
        try:
            self.shape_combo.setCurrentText(m.shape_name)
            self.custom_path_edit.setText(m.custom_path)
            self.voxel_spin.setValue(m.voxel_size)
            self.zoff_spin.setValue(m.z_offset)
            self.phi_spin.setValue(m.phi_offset)

            # colour
            {"class": self.rb_class, "cc": self.rb_cc,
             "solid": self.rb_solid}[m.color_mode].setChecked(True)
            self.class_row_spin.setValue(m.class_row)
            self.solid_color_btn.setRgba(m.solid_color_rgba)
            self.cc_low_color.setRgba(m.lower_cc_color)
            self.cc_high_color.setRgba(m.upper_cc_color)
            self.cc_min_spin.setValue(m.cc_min)
            self.cc_max_spin.setValue(m.cc_max)

            # show
            {objmodel.SHOW_ALL: self.show_all,
             objmodel.SHOW_NONE: self.show_none,
             objmodel.SHOW_ONE: self.show_one,
             objmodel.SHOW_CC: self.show_cc,
             objmodel.SHOW_CLASS: self.show_class}[m.show_mode].setChecked(True)
            n = m.num_particles
            self.one_slider.setRange(1, max(1, n))
            self.one_spin.setRange(1, max(1, n))
            self.one_slider.setValue(min(m.show_one_index, n))
            self.one_spin.setValue(min(m.show_one_index, n))
            lo, hi = m.data_cc_range
            self.cc_low_slider.setRange(lo, hi)
            self.cc_high_slider.setRange(lo, hi)
            self.cc_low_slider.setValue(m.show_cc_low)
            self.cc_high_slider.setValue(m.show_cc_high)
            cmin, cmax = m.class_range()
            self.class_num_spin.setRange(cmin, cmax)
            self.class_num_spin.setValue(
                min(max(m.show_class, cmin), cmax))
        finally:
            self._updating = False

        self._update_enabled_states()

    def _update_enabled_states(self):
        m = self.current_model()
        is_custom = self.shape_combo.currentText() == shapes.CUSTOM_SHAPE
        self.custom_widget.setEnabled(is_custom and m is not None)

        cmode = m.color_mode if m else None
        self.class_row_spin.setEnabled(cmode == "class")
        for w in (self.cc_low_color, self.cc_high_color, self.cc_min_spin,
                  self.cc_max_spin, self.cc_auto_btn):
            w.setEnabled(cmode == "cc")
        self.solid_color_btn.setEnabled(cmode == "solid")

        smode = m.show_mode if m else None
        self.one_w.setEnabled(smode == objmodel.SHOW_ONE)
        self.cc_low_slider.setEnabled(smode == objmodel.SHOW_CC)
        self.cc_high_slider.setEnabled(smode == objmodel.SHOW_CC)
        self.class_num_spin.setEnabled(smode == objmodel.SHOW_CLASS)

    # ===================================================================
    # Control callbacks
    # ===================================================================
    def _shape_changed(self, text):
        self._update_enabled_states()
        if self._updating:
            return
        m = self.current_model()
        if m is None:
            return
        if text == shapes.CUSTOM_SHAPE and not m.custom_path:
            # wait until the user browses for a file
            return
        m.set_shape(text)
        self.status.setText("Object shape: %s" % text)

    def _browse_custom(self):
        m = self.current_model()
        if m is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area, "Choose object STL", "",
            "STL file (*.stl);;All files (*)")
        if not path:
            return
        self.custom_path_edit.setText(path)
        m.set_shape(shapes.CUSTOM_SHAPE, custom_path=path)
        self.status.setText("Custom object: %s" % os.path.basename(path))

    def _placement_changed(self, *args):
        if self._updating:
            return
        m = self.current_model()
        if m is None:
            return
        m.voxel_size = self.voxel_spin.value()
        m.z_offset = self.zoff_spin.value()
        m.phi_offset = self.phi_spin.value()
        m.update_placements()

    def _color_mode_changed(self, checked):
        if self._updating or not checked:
            return
        m = self.current_model()
        if m is None:
            return
        if self.rb_class.isChecked():
            m.color_mode = "class"
        elif self.rb_cc.isChecked():
            m.color_mode = "cc"
        else:
            m.color_mode = "solid"
        self._update_enabled_states()
        m.update_colors()

    def _class_row_changed(self, value):
        if self._updating:
            return
        m = self.current_model()
        if m is None:
            return
        m.class_row = value
        # class range may change; refresh the class show controls
        cmin, cmax = m.class_range()
        self.class_num_spin.setRange(cmin, cmax)
        m.update_colors()
        if m.show_mode == objmodel.SHOW_CLASS:
            m.update_display()

    def _color_changed(self, *args):
        if self._updating:
            return
        m = self.current_model()
        if m is None:
            return
        m.solid_color_rgba = self.solid_color_btn.rgba()
        m.lower_cc_color = self.cc_low_color.rgba()
        m.upper_cc_color = self.cc_high_color.rgba()
        m.cc_min = self.cc_min_spin.value()
        m.cc_max = self.cc_max_spin.value()
        m.update_colors()

    def _auto_cc_range(self):
        m = self.current_model()
        if m is None:
            return
        lo, hi = m.data_cc_range
        self._updating = True
        self.cc_min_spin.setValue(lo)
        self.cc_max_spin.setValue(hi)
        self._updating = False
        self._color_changed()

    def _show_mode_changed(self, checked):
        if self._updating or not checked:
            return
        m = self.current_model()
        if m is None:
            return
        if self.show_all.isChecked():
            m.show_mode = objmodel.SHOW_ALL
        elif self.show_none.isChecked():
            m.show_mode = objmodel.SHOW_NONE
        elif self.show_one.isChecked():
            m.show_mode = objmodel.SHOW_ONE
        elif self.show_cc.isChecked():
            m.show_mode = objmodel.SHOW_CC
        else:
            m.show_mode = objmodel.SHOW_CLASS
        self._update_enabled_states()
        m.update_display()

    def _show_param_changed(self):
        if self._updating:
            return
        m = self.current_model()
        if m is None:
            return
        m.show_one_index = self.one_spin.value()
        m.show_cc_low = self.cc_low_slider.value()
        m.show_cc_high = self.cc_high_slider.value()
        m.show_class = self.class_num_spin.value()
        m.update_display()

    # ===================================================================
    # Actions
    # ===================================================================
    def _focus(self):
        m = self.current_model()
        if m is None:
            return
        from chimerax.core.commands import run
        run(self.session, "view #%s" % m.id_string)

    def _save_shown(self):
        m = self.current_model()
        if m is None:
            return
        default = ""
        if m.source_path:
            default = m.source_path[:-3] + "_display.em" \
                if m.source_path.endswith(".em") else m.source_path + "_display.em"
        path, filt = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Save displayed particles",
            default, "EM motive list (*.em);;STOPGAP star (*.star)")
        if not path:
            return
        fmt = "star" if (path.lower().endswith(".star")
                         or "star" in filt.lower()) else "em"
        n = m.save_displayed(path, fmt=fmt)
        self.status.setText("Saved %d displayed particles to %s"
                            % (n, os.path.basename(path)))

    # ===================================================================
    # Lifecycle
    # ===================================================================
    def close(self):
        if self._remove_handler is not None:
            self.session.triggers.remove_handler(self._remove_handler)
            self._remove_handler = None
