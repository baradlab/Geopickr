# vim: set expandtab ts=4 sw=4:
"""The single tabbed Geopickr tool window (ArtiaX-style).

One dockable window holds three tabs - Place Points, Geometry Picker and Place
Object - each in its own scroll area so a tall panel scrolls internally instead
of pushing the command line and graphics out of view.
"""

from chimerax.core.tools import ToolInstance

from Qt.QtCore import Qt
from Qt.QtWidgets import QVBoxLayout, QTabWidget, QScrollArea


class GeopickrTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:user/tools/pickparticle.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Geopickr"

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        tw.ui_area.setLayout(layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        from .place_points_tool import PlacePointsPanel
        from .pick_tool import GeometryPickerPanel
        from .place_object_tool import PlaceObjectPanel

        self.points_panel = PlacePointsPanel(self)
        self.picker_panel = GeometryPickerPanel(self)
        self.place_object_panel = PlaceObjectPanel(self)

        self._panels = [self.points_panel, self.picker_panel,
                        self.place_object_panel]
        self._add_tab(self.points_panel, "Place Points")
        self._add_tab(self.picker_panel, "Geometry Picker")
        self._add_tab(self.place_object_panel, "Place Object")

        tw.manage(placement="side")

    def _add_tab(self, panel, label):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Fit content to the dock width; never scroll horizontally.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel.widget)
        self.tabs.addTab(scroll, label)

    # -- inter-tab helpers ---------------------------------------------------
    def select_tab(self, panel):
        for i in range(self.tabs.count()):
            if self.tabs.widget(i).widget() is panel.widget:
                self.tabs.setCurrentIndex(i)
                return

    def show_place_object(self, model=None):
        """Refresh the Place Object tab (and raise it) after a pick."""
        self.place_object_panel.refresh()
        self.select_tab(self.place_object_panel)

    def refresh_place_object(self):
        self.place_object_panel.refresh()

    # -- lifecycle -----------------------------------------------------------
    def delete(self):
        for p in self._panels:
            try:
                p.close()
            except Exception:
                pass
        super().delete()

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, "Geopickr",
                                   create=create)
