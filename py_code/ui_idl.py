import os
import random

import numpy as np
from custom import Debug, Dict, DirExplorer, Folder
from custom import Global as g
from custom import Img, Json, List, Time, Value
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QIcon, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget
from str_lib import StrLib as s
from training_idl_gtvn import TrainingIDLGTVn
from ui_replay import UiReplay

# always fill gtvt


class DraggableCross(QWidget):
    def __init__(self, parent, cross_id: int):
        super().__init__(parent)
        self.cross_id = cross_id

        self.__WIDTH = g.CROSS_SIZE
        self.__HEIGHT = g.CROSS_SIZE
        self.setFixedSize(self.__WIDTH, self.__HEIGHT)

        self.setMouseTracking(True)

        self.selected = False
        self.dragging = False
        self.offset = None

        self.png_label = QLabel(self)
        self.png_label.setGeometry(0, 0, self.__WIDTH, self.__HEIGHT)

    def get_pos_in_nii(self):
        rgb_img_relative_pos = self.parent().window().get_rgb_img_relative_pos()
        img_plane = self.parent().window().get_img_plane()
        cur_slice = self.parent().window().get_cur_slice()
        img_shape = self.parent().window().get_3d_img_shape()
        nii_spacing = self.parent().window().get_nii_spacing()

        if rgb_img_relative_pos is None:
            return None

        x = self.pos().x() + round(g.CROSS_SIZE / 2) - rgb_img_relative_pos["x"]
        y = self.pos().y() + round(g.CROSS_SIZE / 2) - rgb_img_relative_pos["y"]

        x = x / rgb_img_relative_pos["width"]
        y = y / rgb_img_relative_pos["height"]

        d, h, w = img_shape

        # 2d to 3d
        if img_plane == s.TRANSVERSE:
            w *= x
            h *= y
            d = cur_slice
        elif img_plane == s.CORONAL:
            w *= x
            h = cur_slice
            d *= y
        elif img_plane == s.SAGITTAL:
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
            self.parent().window().delete_click_in_nii(self)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            new_pos = self.mapToParent(event.pos() - self.offset)
            self.parent().window().move_4_crosses(new_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.parent().window().set_4_crosses_dragging_state(False)
            self.parent().window().add_click_in_nii(self)

    def select(self, selected: bool):
        self.selected = selected
        if selected:
            self.load_png(g.CROSS_DIR_SELECTED)
            # set focus, otherwise key_delete/key_backspace wont work
            self.setFocus()
        else:
            self.load_png(g.CROSS_DIR)

    def load_png(self, png_path: str):
        if os.path.exists(png_path):
            pixmap = QPixmap(png_path)
            pixmap = pixmap.scaled(
                self.__WIDTH, self.__HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.png_label.setPixmap(pixmap)


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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Resize the drawing layer pixmap to match the new size of the QLabel
        self.drawing_layer = self.drawing_layer.scaled(self.size())

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            if self.window().get_cur_patient_idl_step() == s.DRAW_GTVT:
                self.window().draw_on_4_qlabels_press(event)

            elif self.window().get_cur_patient_idl_step() == s.CLICK_GTVN_CENTER:
                self.window().add_4_crosses(event.pos(), add_gtvn_click=True)

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)

        # use event.buttons() instead of event.button()
        # button() returns the mouse button that caused the event, which is Qt::NoButton
        if event.buttons() == Qt.LeftButton:
            if self.window().get_cur_patient_idl_step() == s.DRAW_GTVT:
                self.window().draw_on_4_qlabels_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton:
            if self.window().get_cur_patient_idl_step() == s.DRAW_GTVT:
                self.window().draw_on_4_qlabels_release()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.background_img:
            painter.drawPixmap(self.rect(), self.background_img)
            # 0 for fully transparent, 255 for fully opaque
            painter.setOpacity(100 / 255)
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
            pos.x() - round(g.CROSS_SIZE / 2),
            pos.y() - round(g.CROSS_SIZE / 2),
            g.CROSS_SIZE,
            g.CROSS_SIZE,
        )
        new_cross.load_png(g.CROSS_DIR)
        new_cross.show()
        self.crosses_list.append(new_cross)


class UiIdl(UiReplay):
    def draw_on_4_qlabels_press(self, event: QMouseEvent):
        self.paint_pos = event.pos()

    def draw_on_4_qlabels_move(self, event: QMouseEvent):
        if self.paint_pos is None:
            return
        pen_size = self.get_pen_size()
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            painter = QPainter(self._img_qlabel[i].drawing_layer)

            if self.eraser_mode:
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.setPen(
                    QPen(Qt.transparent, pen_size + 2, Qt.SolidLine, Qt.RoundCap)
                )
            else:
                # smooth
                painter.setRenderHint(QPainter.Antialiasing)
                # Set the composition mode to control alpha blending
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                painter.setPen(
                    QPen(self.pen_color, pen_size, Qt.SolidLine, Qt.RoundCap)
                )

            painter.drawLine(self.paint_pos, event.pos())

            self._img_qlabel[i].update()  # schedule a repaint

        self.paint_pos = event.pos()  # update paint pos

    def draw_on_4_qlabels_release(self):
        self.paint_pos = None

    def __confirm_annotation(self):
        if self.get_cur_patient_idl_step() == s.CLICK_GTVN_CENTER:
            # copy data (dont change origin ndarray)
            idl_gtvn_clicks = self._3d_imgs[s.GTVN_CLICKS].copy()

            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=2)

            # turn upside down
            idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=0)

            # start real idl gtvn
            training_idl_gtvn = TrainingIDLGTVn()
            training_idl_gtvn.real_idl(
                idl_gtvn_id=self._idl_id[s.GTVN],
                patient=self._cur_patient,
                idl_gtvn_clicks=idl_gtvn_clicks,
                dataset_section=self._dataset_section,
                dataset_ver=self._dataset_ver,
            )
            # update idl step for current patient
            self.set_cur_patient_idl_step(s.CORRECTION)

            self._choose_idl_gtvt()
            self._choose_idl_gtvn()
            self._refresh_rgb_imgs()
            self._refresh_title()
            self.__refresh_crosses_on_rgb_imgs()
            self.__save_idl_step()
            self.__update_annotation_msg()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.__refresh_crosses_on_rgb_imgs()

    def _set_img_plane(self):
        super()._set_img_plane()
        self.__refresh_crosses_on_rgb_imgs()

    def __refresh_crosses_on_rgb_imgs(self):
        # remove old crosses
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].delete_all_crosses()

        # draw new crosses based on self.__gtvn_clicks
        if self.get_cur_patient_idl_step() == s.CLICK_GTVN_CENTER:
            img_shape = self.get_3d_img_shape()

            for d, h, w in self.__gtvn_clicks:
                x = y = None
                if self._img_plane == s.TRANSVERSE:
                    if self._cur_slice == d:
                        x = w / img_shape[2]
                        y = h / img_shape[1]

                elif self._img_plane == s.CORONAL:
                    if self._cur_slice == h:
                        x = w / img_shape[2]
                        y = d / img_shape[0]

                elif self._img_plane == s.SAGITTAL:
                    if self._cur_slice == w:
                        x = h / img_shape[1]
                        y = d / img_shape[0]

                # find click on current slice
                if x is not None and y is not None:
                    x *= self._rgb_img_relative_pos["width"]
                    y *= self._rgb_img_relative_pos["height"]
                    x = round(x)
                    y = round(y)
                    x += self._rgb_img_relative_pos["x"]  # - round(g.CROSS_SIZE / 2)
                    y += self._rgb_img_relative_pos["y"]  # - round(g.CROSS_SIZE / 2)
                    self.add_4_crosses(QPoint(x, y), add_gtvn_click=False)

    def delete_click_in_nii(self, cross: DraggableCross):
        pos = cross.get_pos_in_nii()
        self._3d_imgs[s.GTVN_CLICKS][pos[0]][pos[1]][pos[2]] = 0
        self.__gtvn_clicks.remove(pos)
        print("remove:", self.__gtvn_clicks, pos)

    def add_click_in_nii(self, cross: DraggableCross):
        pos = cross.get_pos_in_nii()
        self._3d_imgs[s.GTVN_CLICKS][pos[0]][pos[1]][pos[2]] = 1
        self.__gtvn_clicks.append(pos)
        print("add:", self.__gtvn_clicks, pos)

    def set_4_crosses_dragging_offset(self, pos: QPoint):
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].selected_cross.offset = pos

    def set_4_crosses_dragging_state(self, dragging: bool):
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].selected_cross.dragging = dragging

    def move_4_crosses(self, pos: QPoint):
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].selected_cross.move(pos)

    def delete_4_crosses(self):
        cross = self._img_qlabel[s.CT].selected_cross
        self.delete_click_in_nii(cross)
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].delete_selected_cross()

    # make this function public, CustomQLabel will use it
    def add_4_crosses(self, pos: QPoint, add_gtvn_click: bool):
        if self._3d_imgs[s.CT] is None:
            return

        # make sure new cross id is unique
        crosses_id_list = self._img_qlabel[s.CT].get_crosses_id_list()
        while 1:
            cross_id = random.randint(0, 2**16)
            if cross_id not in crosses_id_list:
                break
        # add crosses
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].add_cross(pos=pos, cross_id=cross_id)

        # add clicks into 3d img
        if add_gtvn_click:
            new_cross = self._img_qlabel[s.CT].get_cross_by_id(cross_id)
            pos = new_cross.get_pos_in_nii()
            self._3d_imgs[s.GTVN_CLICKS][pos[0]][pos[1]][pos[2]] = 1
            self.__gtvn_clicks.append(pos)
            print("add:", self.__gtvn_clicks, pos)

    def select_4_crosses(self, cross_id: int):
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].select_cross(cross_id)

    def get_rgb_img_relative_pos(self):
        return self._rgb_img_relative_pos

    def get_nii_spacing(self):
        return self._nii_spacing

    def get_img_plane(self):
        return self._img_plane

    def get_cur_slice(self):
        return self._cur_slice

    def get_3d_img_shape(self):
        if self._3d_imgs[s.CT] is not None:
            return self._3d_imgs[s.CT].shape
        else:
            return None

    def __init__(
        self,
        idl_remark: str = None,
        debug_mode: bool = False,
    ):
        # pass debug_mode parameter to the parent class
        super().__init__(idl_remark=idl_remark, debug_mode=debug_mode)

    def _init_ui_names(self):
        # before _init_ui_names()
        self._img_qlabel_ct = CustomQLabel(self._central_widget)
        self._img_qlabel_pt = CustomQLabel(self._central_widget)
        self._img_qlabel_mr1 = CustomQLabel(self._central_widget)
        self._img_qlabel_mr2 = CustomQLabel(self._central_widget)

        super()._init_ui_names()

        self._text_label[s.ANNOTATION_TOOLS] = self._text_label_annotation_tools
        self._text_label[s.IDL_PROGRESS] = self._text_label_idl_progress
        self._text_label[s.PEN_SIZE] = self._text_label_pen_size

        self.__btn = Dict()
        self.__btn[s.DRAWING_MODE] = self._btn_drawing_mode
        self.__btn[s.CLEAR] = self._btn_clear
        self.__btn[s.CONFIRM] = self._btn_confirm

    def _init_member_var(self, idl_remark: str = None, debug_mode: bool = False):
        super()._init_member_var()

        # keep idl.gtvt and idl.gtvn id unchanged
        cur_time = Time.get_cur_time_str()
        for i in [s.GTVT, s.GTVN]:
            self._idl_id[i] = "idl.{}_".format(i) + cur_time
            if debug_mode:
                self._idl_id[i] += "_" + s.DELETE_FLAG

            if idl_remark != "" and idl_remark is not None:
                while idl_remark.startswith("_"):
                    idl_remark = idl_remark[1:]
                while idl_remark.endswith("_"):
                    idl_remark = idl_remark[:-1]
                self._idl_id[i] += "_" + idl_remark

        self.__idl_step = Dict()
        for patient in self._patients.to_list():
            self.__idl_step["patient={}".format(patient)] = s.DRAW_GTVT

        # save the position of gtvn clicks
        self.__gtvn_clicks = List()

        # drawing
        self.eraser_mode = False
        self.paint_pos = None  # Store the last painted point

        # r/g/b/transparency, all range from 0 to 255
        self.pen_color = Qt.green  # QColor(0, 255, 0, 150)

    def __save_idl_step(self):
        for i in [s.GTVT, s.GTVN]:
            idl_step_json_path = os.path.join(
                g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i], "idl_step.json"
            )
            Json.save(self.__idl_step, idl_step_json_path)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F12:
            pass

        # delete selected cross
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_4_crosses()

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def __clear_annotation(self):
        # clear drawing layer of each img_qlabel
        for i in [s.CT, s.PT, s.MR1, s.MR2]:
            self._img_qlabel[i].drawing_layer = QPixmap(self.size())
            self._img_qlabel[i].drawing_layer.fill(Qt.transparent)
            self._img_qlabel[i].drawing_layer = self._img_qlabel[
                i
            ].drawing_layer.scaled(self._img_qlabel[i].size())
            self._img_qlabel[i].update()

    def __switch_drawing_mode(self):
        if self.get_cur_patient_idl_step() == s.DRAW_GTVT:
            if self.eraser_mode:
                self.eraser_mode = False
                icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "eraser.png"))
                self.__btn[s.DRAWING_MODE].setIcon(icon)
            else:
                self.eraser_mode = True
                icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "pen.png"))
                self.__btn[s.DRAWING_MODE].setIcon(icon)

    def _init_side_bar(self):
        super()._init_side_bar()

        # hide idl.gtvt/gtvn controls
        for i in [s.BASELINE, s.IDL_GTVT, s.IDL_GTVN]:
            self._text_label[i].hide()
            self._combox[i].hide()
            self._arrow_btn["prev.{}".format(i)].hide()
            self._arrow_btn["next.{}".format(i)].hide()

        # show annotation controls
        self._text_box_annotation_msg.show()
        self._progress_bar_idl.show()
        self._slider_pen_size.show()
        for i in [s.ANNOTATION_TOOLS, s.IDL_PROGRESS, s.PEN_SIZE]:
            self._text_label[i].show()
        for i in [s.DRAWING_MODE, s.CLEAR, s.CONFIRM]:
            self.__btn[i].show()

        # set text
        self._text_box_annotation_msg.setText("Please Select a Patient")
        self._text_label[s.ANNOTATION_TOOLS].setText("Annotation Tools")
        self._text_label[s.IDL_PROGRESS].setText("Retraining Progress")

        # set fonts
        for i in [s.ANNOTATION_TOOLS, s.IDL_PROGRESS, s.PEN_SIZE]:
            self._text_label[i].setFont(self._font_bold)
        self._text_box_annotation_msg.setFont(self._font_bold)

        # set textbox read only
        self._text_box_annotation_msg.setReadOnly(True)

        # pen size slider
        self._slider_pen_size.setMinimum(1)
        self._slider_pen_size.setMaximum(11)
        self._slider_pen_size.setValue(6)

        # set icons
        icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "eraser.png"))
        self.__btn[s.DRAWING_MODE].setIcon(icon)
        icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "clear.png"))
        self.__btn[s.CLEAR].setIcon(icon)
        icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "confirm.png"))
        self.__btn[s.CONFIRM].setIcon(icon)

        # connect ui to functions
        # (put this at the end, because these functions will need the initialization above)
        self.__btn[s.DRAWING_MODE].clicked.connect(self.__switch_drawing_mode)
        self.__btn[s.CLEAR].clicked.connect(self.__clear_annotation)
        self.__btn[s.CONFIRM].clicked.connect(self.__confirm_annotation)

    def get_pen_size(self):
        return self._slider_pen_size.value()

    def _refresh_side_bar(self):
        (
            left,
            top,
            width,
            gap,
            text_height,
            bar_height,
            slider_height,
        ) = super()._refresh_side_bar(widgets_to_display=[s.PATIENT])

        annotation_msg_box_height = 80
        annotation_btn_width = 60

        # annotation tools
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label[s.ANNOTATION_TOOLS].setGeometry(rect)
        self._text_label[s.ANNOTATION_TOOLS].show()
        top += text_height
        tmp_left = left
        annotation_btn_gap = round((width - 3 * annotation_btn_width) / 2)
        for i in [s.DRAWING_MODE, s.CLEAR, s.CONFIRM]:
            rect = QRect(tmp_left, top, annotation_btn_width, bar_height)
            self.__btn[i].setGeometry(rect)
            self.__btn[i].show()
            tmp_left += annotation_btn_gap + annotation_btn_width
        top += bar_height

        # pen size
        rect = QRect(left, top, width, text_height)
        self._text_label[s.PEN_SIZE].setGeometry(rect)
        top += text_height
        rect = QRect(left, top, width, slider_height)
        self._slider_pen_size.setGeometry(rect)
        top += slider_height

        # idl retraining progress bar
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label_idl_progress.setGeometry(rect)
        self._text_label_idl_progress.show()
        top += text_height
        rect = QRect(left, top, width, bar_height)
        self._progress_bar_idl.setGeometry(rect)
        self._progress_bar_idl.show()
        top += bar_height

        # annotation message box
        top += gap
        rect = QRect(left, top, width, annotation_msg_box_height)
        self._text_box_annotation_msg.setGeometry(rect)
        top += annotation_msg_box_height

    def _choose_baseline(self):
        # self._reset_zoomin()
        self._clear_img_data()
        self._clear_img_qlabels()

        self._baseline_id = "baseline_real.idl"
        # self._baseline_id = "baseline_2023.02.27.07.08.09_3mm"

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self._combox[s.PATIENT].setCurrentIndex(-1)  # show nothing

        # # run this after patient combox current text is set up
        # self._enable_arrow_btns(s.PATIENT)

        # create idl folders (after baseline_id is confirmed)
        for i in [s.GTVT, s.GTVN]:
            Folder.create(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i])
            )

    # rewrite this function (do nothing)
    def _add_score_on_rgb_img(self, rgb_img):
        pass

    # rewrite this function (do nothing)
    def _add_label_text_on_rgb_img(self, rgb_img):
        pass

    def _add_pred_text_on_rgb_img(self, rgb_img):
        rgb_img_height = rgb_img.shape[0]
        pos_x = 10
        pos_y = rgb_img_height - 68
        gap_y = 20

        for i in ["t", "n"]:
            if self._3d_imgs["gtv{}.pred".format(i)] is not None:
                cv_text = "GTV{}".format(i)
                pos_y += gap_y
                self._cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(pos_x, pos_y),
                    color=self._color["gtv{}.pred".format(i)],
                )

    def _choose_patient(self, idx: int = None):
        self._cur_patient = self._combox[s.PATIENT].currentText()
        # run these after patient combox current text is set up
        self._enable_arrow_btns(s.PATIENT)
        self._load_dataset_dir_and_nii_spacing()

        # self._reset_zoomin()

        # load multi-modal imgs only, no labels
        self._load_multi_modal_imgs()
        # reset current slice id after ct img loaded
        self._reset_cur_slice_id()
        self._choose_idl_gtvt()
        self._choose_idl_gtvn()
        self._refresh_rgb_imgs()
        self._refresh_title()
        self.__refresh_crosses_on_rgb_imgs()
        self.__save_idl_step()
        self.__update_annotation_msg()

    def _reset_cur_slice_id(self):
        self._cur_slice = self._get_middle_slice_id()

    def get_cur_patient_idl_step(self):
        return self.__idl_step["patient={}".format(self._cur_patient)]

    def set_cur_patient_idl_step(self, step: str):
        self.__idl_step["patient={}".format(self._cur_patient)] = step

    def __update_annotation_msg(self):
        cur_patient_idl_step = self.get_cur_patient_idl_step()

        if cur_patient_idl_step == s.CLICK_GTVT_CENTER:
            self._text_box_annotation_msg.setText(
                "Please click the center of GTVt, then press OK"
            )

        elif cur_patient_idl_step == s.DRAW_GTVT:
            self._text_box_annotation_msg.setText(
                "Please delineate the countour of GTVt on transvers/coronal/sagittal plane, then press OK"
            )

        elif cur_patient_idl_step == s.CLICK_GTVN_CENTER:
            self._text_box_annotation_msg.setText(
                "Please click the center of each involved lymph nodes, then press OK."
            )

        elif cur_patient_idl_step == s.CORRECTION:
            self._text_box_annotation_msg.setText(
                "Please correct the predictions, then press OK"
            )

        else:
            Debug.error_exit("idl step value error")

    def _choose_idl_gtvt(self):
        self.__choose_idl(gtv=s.GTVT)

    def _choose_idl_gtvn(self):
        patient_dir = self.__choose_idl(gtv=s.GTVN)

        gtvn_clicks_nii_path = os.path.join(patient_dir, s.ROUND_01, "gtvn_clicks.nii")
        if os.path.exists(gtvn_clicks_nii_path):
            self._3d_imgs[s.GTVN_CLICKS] = self._load_3d_img(
                path=gtvn_clicks_nii_path, binary=True
            )
        else:
            self._3d_imgs[s.GTVN_CLICKS] = np.zeros(
                self._3d_imgs[s.CT].shape, dtype=np.float32
            )
            # Nii.save(
            #     img=self._3d_imgs[s.GTVN_CLICKS],
            #     save_path=gtvn_clicks_nii_path,
            # )

    def __choose_idl(self, gtv: str) -> str:
        patient_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id[gtv],
            s.PATIENTS,
            "patient={}".format(self._cur_patient),
        )

        # current patient dir exists
        if os.path.exists(patient_dir):
            round_dirs = DirExplorer.get_sub_folders(
                patient_dir, key_word="round=", full_path=True
            )
            # choose the last round
            if len(round_dirs) > 0:
                round_dir = round_dirs[-1]
                pred_path = os.path.join(round_dir, "{}_pred.nii".format(gtv))

                # find idl pred, load it
                if os.path.exists(pred_path):
                    self._3d_imgs["{}.pred".format(gtv)] = Img.binarize(
                        self._load_3d_img(pred_path)
                    )
                # cant find idl pred, clear 3d img
                else:
                    self._3d_imgs["{}.pred".format(gtv)] = None

            # no round dirs found
            else:
                self._3d_imgs["{}.pred".format(gtv)] = None

        # cant find cur patient dir
        else:
            # Folder.create(patient_dir)
            self._3d_imgs["{}.pred".format(gtv)] = None

        return patient_dir
