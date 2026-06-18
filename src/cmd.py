# vim: set expandtab ts=4 sw=4:
"""Command-line interfaces: ``placeobject`` and ``pickparticle``."""

from chimerax.core.commands import (
    CmdDesc, register, OpenFileNameArg, FloatArg, EnumOf, BoolArg, IntArg,
    ModelsArg, ModelArg,
)

from . import shapes


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
                 tomoId=0, onSurface=None, display=True):
    """Geometrically pick particles from markers (or a surface).

    For sphere/tube/filament styles, ``markers`` is a list of marker-set models
    (open a .cmm first, or pass an existing marker set).  For the surface style,
    pass ``onSurface`` (a surface model).
    """
    from . import picking
    from .objmodel import PlacedParticles

    marker_models = list(markers) if markers else []
    motl = picking.pick(
        session, style=style, marker_models=marker_models,
        surface_model=onSurface, radius=radius, tangential=tangential,
        axial=axial, twist=twist, random_phi=randomPhi, tomo_id=tomoId)
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
        ("display", BoolArg),
    ],
    synopsis="Geometrically pick particles from markers",
)


def register_command(command_name, logger):
    if command_name == "placeobject":
        register(command_name, _placeobject_desc, placeobject, logger=logger)
    elif command_name == "pickparticle":
        register(command_name, _pickparticle_desc, pickparticle, logger=logger)
