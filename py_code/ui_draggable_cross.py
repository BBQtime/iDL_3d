import os

from custom import Global as g
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget
from str_lib import DisplayMode, IDLStep, Plane


class DraggableCross(QWidget):
    def __init__(self, parent, cross_id: tuple):
        super().__init__(parent)
        self.cross_id = cross_id

        self.CROSS_SIZE = 20
        self.SELECTED_CROSS_ICON_PATH = os.path.join(
            g.PROJ_DIR, "icons", "cross_selected.png"
        )
        self.UNSELECTED_CROSS_ICON_PATH = os.path.join(
            g.PROJ_DIR, "icons", "cross_unselected.png"
        )

        self.setFixedSize(self.CROSS_SIZE, self.CROSS_SIZE)

        self.setMouseTracking(True)

        self.selected = False
        self.dragging = False
        self.offset = None

        self.png_label = QLabel(self)
        self.png_label.setGeometry(0, 0, self.CROSS_SIZE, self.CROSS_SIZE)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.window().select_cross(self.cross_id)
            self.window().set_crosses_dragging_state(
                img_frame=self.parent(), dragging=True
            )
            self.window().set_crosses_dragging_offset(
                img_frame=self.parent(), pos=event.pos()
            )
            self.window().remove_3d_pos_of_selected_cross(self)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            new_pos = self.mapToParent(event.pos() - self.offset)
            self.window().move_cross(img_frame=self.parent(), pos=new_pos)

    def __get_pos_in_3d(self):
        x = self.pos().x() + round(self.CROSS_SIZE / 2)
        y = self.pos().y() + round(self.CROSS_SIZE / 2)
        return self.parent().get_pos_in_3d(QPoint(x, y))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.window().set_crosses_dragging_state(
                img_frame=self.parent(), dragging=False
            )
            # update cross_id (3d position)
            pos_3d = self.__get_pos_in_3d()
            self.window().update_cross_id(
                cross=self,
                old_cross_id=self.cross_id,
                new_cross_id=pos_3d,
            )
            # add cross id (3d position) into main window
            idl_step = self.window().cur_idl_step()
            if idl_step == IDLStep.CLICK_GTVT_CENTER:
                self.window().gtvt_click_pos_3d = self.cross_id
            elif idl_step == IDLStep.CLICK_GTVN_CENTER:
                self.window().gtvn_clicks_pos_3d.append(self.cross_id)

            # refresh data and img after cross id updated
            self.window().reset_cur_slice_id()
            if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                # only refresh other img frames
                frame_name_list.remove(self.parent().plane)
                for i in frame_name_list:
                    self.window().refresh_imgs(frame_name=i)
                    self.window().refresh_crosses(i)
                # select cross
                self.window().select_cross(self.cross_id)

    def select(self, selected: bool):
        self.selected = selected
        if selected:
            self.load_png(self.SELECTED_CROSS_ICON_PATH)
            # set focus, otherwise key_delete/key_backspace wont work
            self.setFocus()
        else:
            self.load_png(self.UNSELECTED_CROSS_ICON_PATH)

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
