# vim: set expandtab ts=4 sw=4:
"""Geopickr - ChimeraX bundle entry point.

Ports the Briggs-lab subtomogram picking pipeline (Volume Tracer -> Pick
Particle -> Place Object) into one ChimeraX bundle. See bundle_info.xml for
metadata and docs/user/tools/*.html for usage.
"""

from chimerax.core.toolshed import BundleAPI


class _GeopickrAPI(BundleAPI):

    api_version = 1

    # -- Tools ---------------------------------------------------------------
    @staticmethod
    def start_tool(session, bi, ti):
        from .maintool import GeopickrTool
        return GeopickrTool.get_singleton(session)

    # -- Commands ------------------------------------------------------------
    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci.name, logger)

    # -- Session restore -----------------------------------------------------
    @staticmethod
    def get_class(class_name):
        if class_name == "PlacedParticles":
            from .objmodel import PlacedParticles
            return PlacedParticles
        elif class_name == "GeopickrTool":
            from .maintool import GeopickrTool
            return GeopickrTool
        return None


bundle_api = _GeopickrAPI()
