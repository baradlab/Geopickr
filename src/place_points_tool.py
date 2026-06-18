# vim: set expandtab ts=4 sw=4:
"""Place Points tool (Qt): a thin helper over ChimeraX's native marker placement.

This is the ChimeraX equivalent of the old Volume Tracer step. It does not
reimplement marker placement -- it just creates/lists marker sets, activates the
built-in marker mouse modes, and saves .cmm files for the Geometry Picker.
"""

import os

from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
from chimerax.core.commands import run

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QDoubleSpinBox, QFormLayout,
)


def new_marker_set(session, name="markers"):
    """Create and add an empty marker set; return it."""
    from chimerax.markers import MarkerSet
    ms = MarkerSet(session, name)
    session.models.add([ms])
    return ms


class PlacePointsPanel:
    """Helper over ChimeraX's native marker placement. Lives inside a tab."""

    # (button label, mark mode, hint)
    _MODES = (
        ("Mark on plane", "mark plane",
         "click adds a marker on the shown volume plane"),
        ("Mark on surface", "mark surface",
         "click adds a marker on a surface/density"),
        ("Mark free point", "mark point",
         "click adds a marker at the front clip plane"),
    )

    def __init__(self, tool):
        self.tool = tool
        self.session = tool.session
        self.tool_window = tool.tool_window     # shared, used as dialog parent

        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.widget.setLayout(layout)

        # Marker sets
        box = QGroupBox("Marker sets")
        v = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        v.addWidget(self.list)
        row = QHBoxLayout()
        b_new = QPushButton("New marker set")
        b_new.clicked.connect(self._new_set)
        b_open = QPushButton("Open .cmm...")
        b_open.clicked.connect(self._open)
        b_save = QPushButton("Save .cmm...")
        b_save.clicked.connect(self._save)
        for b in (b_new, b_open, b_save):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(box)

        # Placement modes
        mbox = QGroupBox("Place markers (assigns right mouse button)")
        mv = QVBoxLayout(mbox)
        for label, mode, hint in self._MODES:
            r = QHBoxLayout()
            b = QPushButton(label)
            b.clicked.connect(lambda chk=False, m=mode: self._set_mode(m))
            r.addWidget(b)
            r.addWidget(QLabel(hint))
            r.addStretch(1)
            mv.addLayout(r)
        form = QFormLayout()
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 100000.0)
        self.radius_spin.setValue(20.0)
        self.radius_spin.setKeyboardTracking(False)
        self.radius_spin.valueChanged.connect(self._set_radius)
        form.addRow("Marker radius (voxels)", self.radius_spin)
        mv.addLayout(form)
        layout.addWidget(mbox)

        self.status = QLabel(
            "Create a marker set, pick a placement mode, then right-click in the "
            "view. Use the Geometry Picker to turn markers into particles.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._handlers = [
            self.session.triggers.add_handler(ADD_MODELS, lambda *a: self._refresh()),
            self.session.triggers.add_handler(REMOVE_MODELS, lambda *a: self._refresh()),
        ]
        self._refresh()

    def _marker_sets(self):
        from chimerax.markers import MarkerSet
        return [m for m in self.session.models.list(type=MarkerSet)]

    def _refresh(self):
        msets = self._marker_sets()
        cur = self.list.currentRow()
        self.list.clear()
        for m in msets:
            self.list.addItem(QListWidgetItem(
                "#%s  %s  (%d markers)" % (m.id_string, m.name, m.num_atoms)))
        self._models = msets
        if msets and 0 <= cur < len(msets):
            self.list.setCurrentRow(cur)

    def _current(self):
        i = self.list.currentRow()
        if 0 <= i < len(self._models):
            return self._models[i]
        return None

    def _new_set(self):
        ms = new_marker_set(self.session)
        self._refresh()
        self.list.setCurrentRow(self._models.index(ms))
        self.status.setText("Created marker set #%s." % ms.id_string)

    def _set_mode(self, mode):
        run(self.session, 'mousemode right "%s"' % mode)
        self.status.setText("Right mouse: %s. Right-click in the view to add markers." % mode)

    def _set_radius(self, value):
        run(self.session, "marker change markers radius %g" % value, log=False)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area, "Open marker .cmm", "",
            "Chimera markers (*.cmm);;All files (*)")
        if path:
            run(self.session, 'open "%s"' % path)
            self._refresh()

    def _save(self):
        ms = self._current()
        if ms is None:
            self.status.setText("Select a marker set to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Save marker set", ms.name + ".cmm",
            "Chimera markers (*.cmm)")
        if path:
            run(self.session, 'save "%s" models #%s' % (path, ms.id_string))
            self.status.setText("Saved %s" % os.path.basename(path))

    def close(self):
        for h in self._handlers:
            self.session.triggers.remove_handler(h)
        self._handlers = []
