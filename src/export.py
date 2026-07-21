# vim: set expandtab ts=4 sw=4:
"""Export Geopickr motive lists to external STA formats.

Glue between the pure-numpy writers in :mod:`motivelist` and ChimeraX: it pulls
the particle coordinates (ChimeraX scene units) and converts them to each
format's coordinate convention using the source tomogram ``Volume`` (voxel
size / box dimensions / origin), then calls the matching writer.
"""

import os
import numpy as np

from . import motivelist as ml

# format key -> (label, file extension)
FORMATS = {
    "em": ("EM motive list (TOM/AV3)", ".em"),
    "stopgap": ("STOPGAP star", ".star"),
    "dynamo_tbl": ("Dynamo table", ".tbl"),
    "relion5": ("RELION 5 star", ".star"),
    "relion3": ("RELION 3/4 star", ".star"),
}


def list_volumes(session):
    from chimerax.map import Volume
    return [m for m in session.models.list(type=Volume)]


def grid_of(volume):
    """Return (size, step, origin) tuples from a Volume's grid data."""
    d = volume.data
    return tuple(d.size), tuple(d.step), tuple(d.origin)


def scene_to_ijk(volume, pos_scene):
    """Map (N,3) scene-coordinate positions to the volume's voxel indices (float)."""
    p = np.asarray(pos_scene, dtype=np.float64)
    data_xyz = volume.scene_position.inverse() * p     # scene -> volume data coords
    return np.asarray(volume.data.xyz_to_ijk(data_xyz), dtype=np.float64)


def export_model(session, model, path, fmt, volume=None, tomo_id=1,
                 tomo_name=None, vll_path=None, voxel_size=None, dims=None,
                 motl=None, apply_display_offset=True):
    """Write ``model`` (a PlacedParticles) to ``path`` in ``fmt``.

    Coordinates are taken from the motl (ChimeraX scene units) and converted
    using ``volume`` when given, else treated as already in voxels.  Pass
    ``motl`` to export a subset (e.g. only displayed particles); defaults to the
    model's full motl.  When ``apply_display_offset`` is True, the model's Place
    Object Z/phi offset is folded into the exported coordinates.  Returns the
    number of particles written.
    """
    from chimerax.core.errors import UserError
    if fmt not in FORMATS:
        raise UserError("Unknown export format: %s" % fmt)

    if motl is None:
        motl = model.motl
    if apply_display_offset:
        zo = getattr(model, "z_offset", 0.0)
        po = getattr(model, "phi_offset", 0.0)
        if zo or po:
            motl = ml.bake_offsets(motl, z_offset=zo, phi_offset=po)
    n = motl.shape[1]
    if tomo_name is None:
        tomo_name = os.path.splitext(os.path.basename(path))[0]

    # EM and STOPGAP keep the motl's native (scene/voxel) coordinates as-is.
    if fmt == "em":
        ml.write_em_motivelist(path, motl)
        return n
    if fmt == "stopgap":
        # Picks with >= 2 objects (multiple spheres/tubes/filaments, or VTP
        # surface components) split the gold-standard halves by object so whole
        # objects stay together; single-object picks alternate.
        ml.write_stopgap_star(
            path, motl,
            halfset_by_object=getattr(model, "halfset_by_object", False))
        return n

    # Coordinate conversion for voxel/Angstrom formats.
    pos_scene = (motl[ml.ROWS_COORD, :] + motl[ml.ROWS_SHIFT, :]).T   # (N,3)
    if volume is not None:
        size, step, origin = grid_of(volume)
        ijk = scene_to_ijk(volume, pos_scene)
        vox_A = float(step[0]) if voxel_size is None else float(voxel_size)
        box = np.array(size, dtype=np.float64) if dims is None \
            else np.array(dims, dtype=np.float64)
    else:
        ijk = pos_scene                       # assume already in voxels
        vox_A = float(voxel_size) if voxel_size else 1.0
        box = np.array(dims, dtype=np.float64) if dims is not None else None

    if fmt == "dynamo_tbl":
        pos_int1 = np.round(ijk) + 1.0
        shifts = ijk - np.round(ijk)
        ml.write_dynamo_tbl(path, motl, pos_int1, shifts, tomo_id=int(tomo_id))
        attached = None
        if vll_path:
            attached = ml.append_vll_table_ref(vll_path, path, tomo_substr=tomo_name)
        if attached is not None:
            session.logger.info("Geopickr: referenced %s in %s (tomogram #%d)"
                                % (os.path.basename(path),
                                   os.path.basename(vll_path), attached))
        return n

    if fmt == "relion3":
        ml.write_relion3_star(path, motl, ijk, tomo_name=tomo_name)
        return n

    if fmt == "relion5":
        if box is None:
            raise UserError(
                "RELION 5 export needs box dimensions: choose the tomogram "
                "Volume, or set dimensions manually.")
        center = box / 2.0          # RELION 5 centers at tomo_size/2 (ArtiaX convention)
        centered_A = (ijk - center) * vox_A
        ml.write_relion5_star(path, motl, centered_A, tomo_name=tomo_name)
        return n

    raise UserError("Unhandled format: %s" % fmt)


# ---------------------------------------------------------------------------
# Shared Export dialog (Qt)
# ---------------------------------------------------------------------------
def run_export_dialog(tool, model, motl=None):
    """Open the Export dialog for ``model`` and perform the export. GUI only.

    Pass ``motl`` to export a subset (e.g. only the displayed particles).
    """
    from Qt.QtWidgets import (
        QDialog, QFormLayout, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
        QHBoxLayout, QPushButton, QWidget, QDialogButtonBox, QFileDialog, QLabel,
        QCheckBox,
    )
    session = tool.session
    parent = tool.tool_window.ui_area

    dlg = QDialog(parent)
    dlg.setWindowTitle("Export particles")
    form = QFormLayout(dlg)

    fmt_combo = QComboBox()
    fmt_keys = ["em", "stopgap", "dynamo_tbl", "relion5", "relion3"]
    for k in fmt_keys:
        fmt_combo.addItem(FORMATS[k][0], k)
    fmt_combo.setCurrentIndex(fmt_keys.index("dynamo_tbl"))
    form.addRow("Format", fmt_combo)

    vol_combo = QComboBox()
    volumes = list_volumes(session)
    vol_combo.addItem("(none — coords already in voxels)", None)
    for v in volumes:
        vol_combo.addItem("#%s  %s" % (v.id_string, v.name), v)
    if volumes:
        vol_combo.setCurrentIndex(1)
    form.addRow("Tomogram", vol_combo)

    vox_spin = QDoubleSpinBox()
    vox_spin.setRange(0.0, 1e6)
    vox_spin.setDecimals(3)
    vox_spin.setSpecialValueText("auto")
    vox_spin.setValue(0.0)
    form.addRow("Voxel size (Å, manual)", vox_spin)

    tomo_id_spin = QSpinBox()
    tomo_id_spin.setRange(1, 1000000)
    form.addRow("Tomogram ID (Dynamo)", tomo_id_spin)

    name_edit = QLineEdit()
    name_edit.setPlaceholderText("defaults to output filename")
    form.addRow("Tomogram name", name_edit)

    vll_row = QHBoxLayout()
    vll_edit = QLineEdit()
    vll_edit.setPlaceholderText("optional .vll to append '> table' (Dynamo)")
    vll_btn = QPushButton("…")
    vll_row.addWidget(vll_edit, 1)
    vll_row.addWidget(vll_btn)
    vll_w = QWidget()
    vll_w.setLayout(vll_row)
    vll_row.setContentsMargins(0, 0, 0, 0)
    form.addRow(".vll file", vll_w)

    def browse_vll():
        p, _ = QFileDialog.getOpenFileName(dlg, "Choose Dynamo .vll", "",
                                           "Dynamo volume list (*.vll);;All files (*)")
        if p:
            vll_edit.setText(p)
    vll_btn.clicked.connect(browse_vll)

    # Bake the Place Object display Z/phi offset into the exported coordinates.
    zo = getattr(model, "z_offset", 0.0)
    po = getattr(model, "phi_offset", 0.0)
    offset_check = QCheckBox("Bake Place Object Z/phi offset into coordinates")
    offset_check.setChecked(True)
    offset_check.setEnabled(bool(zo or po))
    lbl = ("Z offset %g, phi %g" % (zo, po)) if (zo or po) else "none set"
    form.addRow("Display offset", QLabel(lbl))
    form.addRow("", offset_check)

    bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    form.addRow(bb)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)

    if dlg.exec() != QDialog.Accepted:
        return None

    fmt = fmt_combo.currentData()
    volume = vol_combo.currentData()
    voxel_size = vox_spin.value() or None
    tomo_id = tomo_id_spin.value()
    tomo_name = name_edit.text().strip() or None
    vll_path = vll_edit.text().strip() or None
    apply_offset = offset_check.isChecked()

    ext = FORMATS[fmt][1]
    default = (os.path.splitext(model.source_path)[0] if model.source_path
              else model.name.split(" ")[0]) + ext
    path, _ = QFileDialog.getSaveFileName(parent, "Save %s" % FORMATS[fmt][0],
                                          default, "*%s" % ext)
    if not path:
        return None
    try:
        count = export_model(session, model, path, fmt, volume=volume,
                             tomo_id=tomo_id, tomo_name=tomo_name,
                             vll_path=vll_path, voxel_size=voxel_size, motl=motl,
                             apply_display_offset=apply_offset)
    except Exception as e:
        session.logger.warning("Geopickr export: %s" % e)
        return ("error", str(e))
    return ("ok", "Exported %d particles to %s" % (count, os.path.basename(path)))
