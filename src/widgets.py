# vim: set expandtab ts=4 sw=4:
"""Small reusable Qt widgets for the Place Object GUI."""

from Qt.QtCore import Qt, Signal
from Qt.QtWidgets import (
    QWidget, QHBoxLayout, QSlider, QDoubleSpinBox, QPushButton,
)
from Qt.QtGui import QColor


class FloatSlider(QWidget):
    """A horizontal slider coupled to a double spin box, over a float range."""

    valueChanged = Signal(float)

    def __init__(self, decimals=4, steps=1000, parent=None):
        super().__init__(parent)
        self._lo, self._hi, self._steps = 0.0, 1.0, steps
        self._block = False

        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, steps)
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setKeyboardTracking(False)
        self._spin.setMaximumWidth(90)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._spin)
        self.setLayout(lay)

        self._slider.valueChanged.connect(self._slider_changed)
        self._spin.valueChanged.connect(self._spin_changed)

    def setRange(self, lo, hi):
        self._lo, self._hi = float(lo), float(hi)
        self._block = True
        self._spin.setRange(self._lo, self._hi)
        span = self._hi - self._lo
        self._spin.setSingleStep(span / self._steps if span > 0 else 0.01)
        self._block = False

    def _to_slider(self, v):
        span = self._hi - self._lo
        if span <= 0:
            return 0
        return int(round((v - self._lo) / span * self._steps))

    def _to_value(self, s):
        span = self._hi - self._lo
        return self._lo + (s / self._steps) * span

    def setValue(self, v):
        self._block = True
        self._spin.setValue(v)
        self._slider.setValue(self._to_slider(v))
        self._block = False

    def value(self):
        return self._spin.value()

    def _slider_changed(self, s):
        if self._block:
            return
        self._block = True
        v = self._to_value(s)
        self._spin.setValue(v)
        self._block = False
        self.valueChanged.emit(v)

    def _spin_changed(self, v):
        if self._block:
            return
        self._block = True
        self._slider.setValue(self._to_slider(v))
        self._block = False
        self.valueChanged.emit(v)


class ColorButton(QPushButton):
    """A button that shows and edits an RGBA colour via a colour dialog."""

    colorChanged = Signal(tuple)

    def __init__(self, rgba=(128, 128, 128, 255), parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 22)
        self._rgba = tuple(rgba)
        self._refresh()
        self.clicked.connect(self._pick)

    def rgba(self):
        return self._rgba

    def setRgba(self, rgba):
        self._rgba = tuple(rgba)
        self._refresh()

    def _refresh(self):
        r, g, b, a = self._rgba
        self.setStyleSheet(
            "background-color: rgba(%d,%d,%d,%d); border: 1px solid #555;"
            % (r, g, b, a))

    def _pick(self):
        from Qt.QtWidgets import QColorDialog
        r, g, b, a = self._rgba
        c = QColorDialog.getColor(
            QColor(r, g, b, a), self, "Select colour",
            QColorDialog.ShowAlphaChannel)
        if c.isValid():
            self._rgba = (c.red(), c.green(), c.blue(), c.alpha())
            self._refresh()
            self.colorChanged.emit(self._rgba)
