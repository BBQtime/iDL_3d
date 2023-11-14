from custom import IDLStep
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QLabel
from ui_draggable_cross import DraggableCross


class CustomQLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        # clicks
        self.selected_cross = None
        self.crosses_list = []

        # gtvt painting
        self.setMouseTracking(True)
        self.background_img = None
        self.drawing_layer = QPixmap(self.size())
        self.drawing_layer.fill(Qt.transparent)
        self.pen_mode = True

    def clear_drawing_layer(self):
        self.drawing_layer = QPixmap(self.size())
        self.drawing_layer.fill(Qt.transparent)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Resize the drawing layer pixmap to match the new size of the QLabel
        self.drawing_layer = self.drawing_layer.scaled(self.size())

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            idl_step = self.window().get_cur_patient_idl_step()

            if idl_step is None:
                return

            elif idl_step == IDLStep.CLICK_GTVT_CENTER:
                # remove old crosses
                self.window().delete_all_crosses_on_4_qlabels()
                self.window().clear_gtvt_click_pos_3d()
                # add new crosses
                self.window().add_4_crosses(event.pos(), record_click_pos=True)

            elif idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
                self.window().draw_on_4_qlabels_press(event)

            elif idl_step == IDLStep.CLICK_GTVN_CENTER:
                self.window().add_4_crosses(event.pos(), record_click_pos=True)

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)

        # use event.buttons() instead of event.button()
        # button() returns the mouse button that caused the event, which is Qt::NoButton
        if event.buttons() == Qt.LeftButton:
            idl_step = self.window().get_cur_patient_idl_step()

            if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
                self.window().draw_on_4_qlabels_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton:
            idl_step = self.window().get_cur_patient_idl_step()

            if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
                self.window().draw_on_4_qlabels_release()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.background_img:
            painter.setOpacity(1.0)
            painter.drawPixmap(self.rect(), self.background_img)

        if self.drawing_layer:
            # 0 for fully transparent, 255 for fully opaque
            if self.pen_mode:
                painter.setOpacity(130 / 255)
            else:
                painter.setOpacity(180 / 255)
            painter.drawPixmap(self.rect(), self.drawing_layer)

    def set_background(self, img: QImage):
        self.background_img = QPixmap.fromImage(img)
        self.update()

    def get_cross_by_id(self, cross_id: int) -> DraggableCross:
        for cross in self.crosses_list:
            if cross.cross_id == cross_id:
                return cross
        # no id found, return None
        return None

    def get_crosses_id_list(self) -> list:
        crosses_id_list = []
        for cross in self.crosses_list:
            crosses_id_list.append(cross.cross_id)
        return crosses_id_list

    def deselect_cross(self):
        if self.selected_cross:
            self.selected_cross.select(False)
            self.selected_cross = None

    def select_cross(self, cross_id: int):
        self.deselect_cross()
        # select new cross
        for cross in self.crosses_list:
            if cross.cross_id == cross_id:
                self.selected_cross = cross
                self.selected_cross.select(True)

    def delete_all_crosses(self):
        for cross in self.crosses_list:
            cross.setParent(None)
            cross.deleteLater()
        self.crosses_list = []
        self.selected_cross = None

    def delete_selected_cross(self):
        if self.selected_cross:
            self.crosses_list.remove(self.selected_cross)
            self.selected_cross.setParent(None)
            self.selected_cross.deleteLater()
            self.selected_cross = None

    def add_cross(self, pos: QPoint, cross_id: int):
        self.deselect_cross()
        # create new cross
        new_cross = DraggableCross(parent=self, cross_id=cross_id)
        new_cross.setGeometry(
            pos.x() - round(new_cross.CROSS_SIZE / 2),
            pos.y() - round(new_cross.CROSS_SIZE / 2),
            new_cross.CROSS_SIZE,
            new_cross.CROSS_SIZE,
        )
        new_cross.load_png(new_cross.CROSS_ICON_DIR_UNSELECTED)
        new_cross.show()
        self.crosses_list.append(new_cross)
