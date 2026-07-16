# vim: set expandtab ts=4 sw=4:
"""Command-line interfaces: ``placeobject``, ``pickparticle``, ``geopickr export``."""

from chimerax.core.commands import (
    CmdDesc, register, OpenFileNameArg, SaveFileNameArg, FloatArg, EnumOf,
    BoolArg, IntArg, StringArg, ModelsArg, ModelArg,
)

from . import shapes

# command fmt name -> export.export_model fmt key
_EXPORT_FMTS = {"em": "em", "stopgap": "stopgap", "dynamoTbl": "dynamo_tbl",
                "relion5": "relion5", "relion3": "relion3"}


# ---------------------------------------------------------------------------
# placeobject
# ---------------------------------------------------------------------------
def placeobject(session, file, shape="Sphere", voxelSize=1.0,
                zOffset=0.0, phiOffset=0.0):
    """Open a motive list and place objects at every particle."""
    from .objmodel import PlacedParticles
    from . import motivelist as ml
    import os

    motl = ml.read_em_motivelist(file)
    name = os.path.basename(file)
    model = PlacedParticles(session, name, motl, source_path=file,
                            shape_name=shape)
    model.voxel_size = voxelSize
    model.z_offset = zOffset
    model.phi_offset = phiOffset
    model.update_placements()
    session.models.add([model])

    _refresh_tool(session)
    session.logger.info("Place Object: loaded %s (%d particles)"
                        % (name, model.num_particles))
    return model


def _refresh_tool(session):
    if getattr(session, "ui", None) is not None and session.ui.is_gui:
        from .maintool import GeopickrTool
        tool = GeopickrTool.get_singleton(session)
        if tool is not None:
            tool.refresh_place_object()


_placeobject_desc = CmdDesc(
    required=[("file", OpenFileNameArg)],
    keyword=[
        ("shape", EnumOf(list(shapes.BUILTIN_SHAPES))),
        ("voxelSize", FloatArg),
        ("zOffset", FloatArg),
        ("phiOffset", FloatArg),
    ],
    synopsis="Place geometric objects at subtomogram positions from a motive list",
)


# ---------------------------------------------------------------------------
# pickparticle
# ---------------------------------------------------------------------------
def pickparticle(session, markers=None, style="sphere", radius=20.0,
                 tangential=0.0, axial=0.0, twist=0.0, randomPhi=None,
                 tomoId=0, onSurface=None, offset=0.0, display=True):
    """Geometrically pick particles from markers (or a surface).

    For sphere/tube/filament styles, ``markers`` is a list of marker-set models
    (open a .cmm first, or pass an existing marker set).  For the surface style,
    pass ``onSurface`` (a surface model).
    """
    from . import picking
    from .objmodel import PlacedParticles
    from chimerax.core.errors import UserError

    marker_models = list(markers) if markers else []
    if style.lower() == "surface":
        if onSurface is None:
            raise UserError("No surface was specified (use onSurface #model).")
    elif not marker_models:
        raise UserError("No markers were specified (give a marker-set model, "
                        "e.g. after 'open markers.cmm').")

    motl = picking.pick(
        session, style=style, marker_models=marker_models,
        surface_model=onSurface, radius=radius, tangential=tangential,
        axial=axial, twist=twist, random_phi=randomPhi, tomo_id=tomoId,
        offset=offset)
    if motl.shape[1] == 0:
        session.logger.warning("Pick Particle: no particles generated.")
        return None

    name = "%s (%s)" % (
        marker_models[0].name if marker_models
        else (onSurface.name if onSurface is not None else "picked"), style)
    model = PlacedParticles(session, name, motl, shape_name="Hexagon")
    model.voxel_size = 0.1
    model.update_placements()
    if display:
        session.models.add([model])

    _refresh_tool(session)
    session.logger.info("Pick Particle: %d particles (%s)"
                        % (model.num_particles, style))
    return model


_pickparticle_desc = CmdDesc(
    optional=[("markers", ModelsArg)],
    keyword=[
        ("style", EnumOf(["sphere", "tube", "filament", "surface"])),
        ("radius", FloatArg),
        ("tangential", FloatArg),
        ("axial", FloatArg),
        ("twist", FloatArg),
        ("randomPhi", BoolArg),
        ("tomoId", IntArg),
        ("onSurface", ModelArg),
        ("offset", FloatArg),
        ("display", BoolArg),
    ],
    synopsis="Geometrically pick particles from markers",
)


# ---------------------------------------------------------------------------
# geopickr export
# ---------------------------------------------------------------------------
def geopickr_export(session, model, file, format="dynamoTbl", onTomogram=None,
                    tomoId=1, tomoName=None, vll=None, voxelSize=None,
                    applyOffset=True):
    """Export a placed-particle model to em / stopgap / Dynamo / RELION files."""
    from chimerax.core.errors import UserError
    from .objmodel import PlacedParticles
    from . import export

    if not isinstance(model, PlacedParticles):
        raise UserError("model must be a Geopickr particle model (PlacedParticles).")
    fmt = _EXPORT_FMTS.get(format)
    if fmt is None:
        raise UserError("Unknown format %r" % format)

    count = export.export_model(
        session, model, file, fmt, volume=onTomogram, tomo_id=tomoId,
        tomo_name=tomoName, vll_path=vll, voxel_size=voxelSize,
        apply_display_offset=applyOffset)
    import os
    session.logger.info("Geopickr export: %d particles -> %s (%s)"
                        % (count, os.path.basename(file), format))


_geopickr_export_desc = CmdDesc(
    required=[("model", ModelArg)],
    keyword=[
        ("file", SaveFileNameArg),
        ("format", EnumOf(list(_EXPORT_FMTS))),
        ("onTomogram", ModelArg),
        ("tomoId", IntArg),
        ("tomoName", StringArg),
        ("vll", OpenFileNameArg),
        ("voxelSize", FloatArg),
        ("applyOffset", BoolArg),
    ],
    required_arguments=["file"],
    synopsis="Export a Geopickr particle model to Dynamo/RELION/EM formats",
)


def register_command(command_name, logger):
    if command_name == "placeobject":
        register(command_name, _placeobject_desc, placeobject, logger=logger)
    elif command_name == "pickparticle":
        register(command_name, _pickparticle_desc, pickparticle, logger=logger)
    elif command_name == "geopickr export":
        register(command_name, _geopickr_export_desc, geopickr_export, logger=logger)
