# vim: set expandtab ts=4 sw=4:
"""Built-in object shapes and STL geometry loading."""

import os

# Built-in shapes shipped with the bundle, in the order shown in the menu.
BUILTIN_SHAPES = (
    "Sphere", "ArrowX", "ArrowY", "ArrowZ", "Rectangle",
    "Triangle", "Square", "Pentagon", "Hexagon",
)

CUSTOM_SHAPE = "Custom STL..."

_OBJECTS_DIR = os.path.join(os.path.dirname(__file__), "objects")


def builtin_path(shape_name):
    """Return the bundled STL path for a built-in shape name."""
    return os.path.join(_OBJECTS_DIR, shape_name + ".stl")


def load_geometry(path):
    """Load an STL file and return (vertices, normals, triangles) numpy arrays.

    Uses ChimeraX's own STL reader so both binary and ASCII files work.
    """
    from chimerax.stl.stl import (
        stl_is_ascii, read_ascii_stl_geometry, read_binary_stl_geometry,
    )
    if stl_is_ascii(path):
        return read_ascii_stl_geometry(path)
    return read_binary_stl_geometry(path)
