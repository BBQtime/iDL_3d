import os

from custom import Global as g
from custom import Value
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget
from str_lib import CORONAL, SAGITTAL, TRANSVERSE


class DraggableCross(QWidget):
    def __init__(self, parent, cross_id: int):
        super().__init__(parent)
        self.cross_id = cross_id

        self.CROSS_SIZE = 20
        self.CROSS_ICON_DIR_SELECTED = os.path.join(
            g.PROJ_DIR, "icons", "cross_selected.png"
        )
        self.CROSS_ICON_DIR_UNSELECTED = os.path.join(
            g.PROJ_DIR, "icons", "cross_unselected.png"
        )

        self.setFixedSize(self.CROSS_SIZE, self.CROSS_SIZE)

        self.setMouseTracking(True)

        self.selected = False
        self.dragging = False
        self.offset = None

        self.png_label = QLabel(self)
        self.png_label.setGeometry(0, 0, self.CROSS_SIZE, self.CROSS_SIZE)

    def get_pos_in_3d(self):
        rgb_img_roi = self.parent().window().get_rgb_img_roi()
        img_plane = self.parent().window().get_img_plane()
        cur_slice = self.parent().window().get_cur_slice()
        img_shape = self.parent().window().get_3d_img_shape()
        nii_spacing = self.parent().window().get_nii_spacing()

        if rgb_img_roi is None:
            return None

        x = self.pos().x() + round(self.CROSS_SIZE / 2) - rgb_img_roi["x"]
        y = self.pos().y() + round(self.CROSS_SIZE / 2) - rgb_img_roi["y"]

        x = x / rgb_img_roi["width"]
        y = y / rgb_img_roi["height"]

        d, h, w = img_shape

        # 2d to 3d
        if img_plane == TRANSVERSE:
            w *= x
            h *= y
            d = cur_slice
        elif img_plane == CORONAL:
            w *= x
            h = cur_slice
            d *= y
        elif img_plane == SAGITTAL:
            w = cur_slice
            h *= x
            d *= y

        w = round(w)
        h = round(h)
        d = round(d)
        w = Value.limit_range(w, (0, img_shape[2] - 1))
        h = Value.limit_range(h, (0, img_shape[1] - 1))
        d = Value.limit_range(d, (0, img_shape[0] - 1))

        # dont neet to turn upside down
        # d = img_shape[0] - d

        # flip left/right back for 1mm data
        if nii_spacing == 1.0:
            w = img_shape[2] - w

        return d, h, w

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.parent().window().select_4_crosses(self.cross_id)
            self.parent().window().set_4_crosses_dragging_state(True)
            self.parent().window().set_4_crosses_dragging_offset(event.pos())
            self.parent().window().delete_click_pos(self)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            new_pos = self.mapToParent(event.pos() - self.offset)
            self.parent().window().move_4_crosses(new_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.parent().window().set_4_crosses_dragging_state(False)
            self.parent().window().add_click_pos(self)

    def select(self, selected: bool):
        self.selected = selected
        if selected:
            self.load_png(self.CROSS_ICON_DIR_SELECTED)
            # set focus, otherwise key_delete/key_backspace wont work
            self.setFocus()
        else:
            self.load_png(self.CROSS_ICON_DIR_UNSELECTED)

    def load_png(self, png_path: str):
        if os.path.exists(png_path):
            pixmap = QPixmap(png_path)
            pixmap = pixmap.scaled(
                self.CROSS_SIZE,
                self.CROSS_SIZE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.png_label.setPixmap(pixmap)
