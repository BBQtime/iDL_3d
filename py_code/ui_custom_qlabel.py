from custom import Dict, Value
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QLabel
from str_lib import CORONAL, CT, SAGITTAL, TRANSVERSE, DisplayMode, IDLStep
from ui_draggable_cross import DraggableCross


class ROI:
    x = None
    y = None
    width = None
    height = None


class CustomQLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        # record plane and modality here instead of main window
        self.plane = None
        self.modal = None

        # rgb img - region of interest
        self.roi = ROI()

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
                self.window().delete_all_crosses()
                # add new 3d pos
                pos_3d = self.get_pos_in_3d(event.pos())
                self.window().gtvt_click_pos_3d = pos_3d
                self.window().reset_cur_slice_id()
                if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                    # only refresh other img_qlabels
                    img_name_list = [TRANSVERSE, CORONAL, SAGITTAL]
                    img_name_list.remove(self.plane)
                    for i in img_name_list:
                        self.window().refresh_img_qlabels(i)
                self.window().refresh_crosses_on_qlabels()

            elif idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
                self.window().draw_on_img_qlabels_press(event)

            elif idl_step == IDLStep.CLICK_GTVN_CENTER:
                pos_3d = self.get_pos_in_3d(event.pos())
                if pos_3d not in self.window().gtvn_clicks_pos_3d:
                    self.window().gtvn_clicks_pos_3d.append(pos_3d)
                    self.window().reset_cur_slice_id()
                    if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                        # only refresh other img_qlabels
                        img_name_list = [TRANSVERSE, CORONAL, SAGITTAL]
                        img_name_list.remove(self.plane)
                        for i in img_name_list:
                            self.window().refresh_img_qlabels(i)
                    self.window().refresh_crosses_on_qlabels()

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)

        # use event.buttons() instead of event.button()
        # button() returns the mouse button that caused the event, which is Qt::NoButton
        if event.buttons() == Qt.LeftButton:
            idl_step = self.window().get_cur_patient_idl_step()

            if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
                self.window().draw_on_img_qlabels_move(event=event, img_qlabel=self)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton:
            idl_step = self.window().get_cur_patient_idl_step()

            if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
                self.window().draw_on_img_qlabels_release(self)

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

    def select_cross(self, cross_id: tuple):
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
            self.window().remove_3d_pos_of_selected_cross(self.selected_cross)
            self.crosses_list.remove(self.selected_cross)
            self.selected_cross.setParent(None)
            self.selected_cross.deleteLater()
            self.selected_cross = None

    def get_pos_in_3d(self, pos: QPoint):
        if self.roi.x is None:
            return

        cur_slice = self.window().cur_slice_id[self.plane]
        img_shape_3d = self.window().get_3d_img_shape()

        x = pos.x() - self.roi.x
        y = pos.y() - self.roi.y

        x = x / self.roi.width
        y = y / self.roi.height

        d, h, w = img_shape_3d

        # 2d to 3d
        if self.plane == TRANSVERSE:
            w *= x
            h *= y
            d = cur_slice
        elif self.plane == CORONAL:
            w *= x
            h = cur_slice
            d *= y
        elif self.plane == SAGITTAL:
            w = cur_slice
            h *= x
            d *= y

        w = round(w)
        h = round(h)
        d = round(d)
        w = Value.limit_range(w, (0, img_shape_3d[2] - 1))
        h = Value.limit_range(h, (0, img_shape_3d[1] - 1))
        d = Value.limit_range(d, (0, img_shape_3d[0] - 1))

        # dont neet to turn upside down
        # flip left/right back for 1mm data
        if self.window().get_nii_spacing() == 1.0:
            w = img_shape_3d[2] - w

        return d, h, w

    def refresh_crosses(self):
        idl_step = self.window().get_cur_patient_idl_step()

        if idl_step not in [IDLStep.CLICK_GTVT_CENTER, IDLStep.CLICK_GTVN_CENTER]:
            return

        # remove old crosses
        self.delete_all_crosses()

        # load crosses position from gtvt/gtvn_clicks_pos_3d
        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            if self.window().gtvt_click_pos_3d is None:
                return
            else:
                clicks_pos_3d = [self.window().gtvt_click_pos_3d]

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            if len(self.window().gtvn_clicks_pos_3d) <= 0:
                return
            else:
                clicks_pos_3d = self.window().gtvn_clicks_pos_3d

        img_shape = self.window().get_3d_img_shape()

        # loop through all clicks
        for d, h, w in clicks_pos_3d:
            x = y = None
            if self.plane == TRANSVERSE and self.window().cur_slice_id[TRANSVERSE] == d:
                x = w / img_shape[2]
                y = h / img_shape[1]

            elif self.plane == CORONAL and self.window().cur_slice_id[CORONAL] == h:
                x = w / img_shape[2]
                y = d / img_shape[0]

            elif self.plane == SAGITTAL and self.window().cur_slice_id[SAGITTAL] == w:
                x = h / img_shape[1]
                y = d / img_shape[0]

            # find click on current slice
            if x is not None and y is not None:
                x *= self.roi.width
                y *= self.roi.height
                x = round(x)
                y = round(y)
                x += self.roi.x
                y += self.roi.y

                # do not record click pos when refreshing
                self.add_cross(pos=QPoint(x, y), click_pos_3d=(d, h, w))

    def add_cross(self, pos: QPoint, click_pos_3d: tuple):
        self.deselect_cross()
        # create new cross (use click_pos_3d as cross_id)
        new_cross = DraggableCross(parent=self, cross_id=click_pos_3d)
        new_cross.setGeometry(
            pos.x() - round(new_cross.CROSS_SIZE / 2),
            pos.y() - round(new_cross.CROSS_SIZE / 2),
            new_cross.CROSS_SIZE,
            new_cross.CROSS_SIZE,
        )
        new_cross.load_png(new_cross.UNSELECTED_CROSS_ICON_PATH)
        new_cross.show()
        self.crosses_list.append(new_cross)

    def wheelEvent(self, event):
        super().wheelEvent(event)

        ct_img = self.window().img_3d[CT]
        if self.plane == SAGITTAL:
            slices_count = ct_img.shape[2]
        elif self.plane == CORONAL:
            slices_count = ct_img.shape[1]
        elif self.plane == TRANSVERSE:
            slices_count = ct_img.shape[0]

        if slices_count == 0:
            return

        slice_delta = event.angleDelta().y() // 120
        if self.plane == CORONAL:
            slice_delta = -slice_delta

        self.window().cur_slice_id[self.plane] -= slice_delta
        # limite slice_id in range (0, slices_count)
        self.window().cur_slice_id[self.plane] %= slices_count

        if self.window().display_mode() == DisplayMode.PLANE_FIXED:
            self.window().refresh_img_qlabels(img_name=self.plane)
        else:
            self.window().refresh_img_qlabels()
