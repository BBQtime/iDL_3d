import os

import cv2
import numpy as np
from custom import Debug, Dict, Directory, Folder
from custom import Global as g
from custom import Img, Json, List, Nii, Time, Value
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QImage, QKeyEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget
from ui_replay import UiReplay

# idl step
CHOOSE_PATIENT = "choose.patient"
CLICK_GTVT_CENTER = "click.gtvt.center"
DELINEATE_GTVT = "delineate.gtvt"
CLICK_GTVN_CENTER = "click.gtvn.center"
CORRECTION = "correction"

# icon path
CROSS_DIR_SELECTED = os.path.join(g.PROJ_DIR, "icons", "cross_selected.png")
CROSS_DIR = os.path.join(g.PROJ_DIR, "icons", "cross.png")


# 2.select cross_png on 4 modals

# 4.record cross_pos into json file

# 5.refresh cross_png when select a new slice


class DraggableCross(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.__WIDTH = 20
        self.__HEIGHT = 20

        self.setFixedSize(self.__WIDTH, self.__HEIGHT)
        self.setMouseTracking(True)
        self.dragging = False
        self.selected = False

        self.png_label = QLabel(self)
        self.png_label.setGeometry(0, 0, self.__WIDTH, self.__HEIGHT)

    def mousePressEvent(self, event: QKeyEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()
            self.parent().select_cross(self)

    def mouseMoveEvent(self, event):
        if self.dragging:
            new_pos = self.mapToParent(event.pos() - self.offset)
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def select(self, selected):
        self.selected = selected
        if selected:
            self.load_png(CROSS_DIR_SELECTED)
            # set focus, otherwise key_delete/key_backspace wont work
            self.setFocus()
        else:
            self.load_png(CROSS_DIR)

    def load_png(self, png_path):
        if os.path.exists(png_path):
            pixmap = QPixmap(png_path)
            pixmap = pixmap.scaled(
                self.__WIDTH, self.__HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.png_label.setPixmap(pixmap)


class CustomQLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_cross = None

    def select_cross(self, cross):
        if self.selected_cross:
            self.selected_cross.select(False)
        self.selected_cross = cross
        if self.selected_cross:
            self.selected_cross.select(True)

    def delete_selected_cross(self):
        if self.selected_cross:
            self.selected_cross.setParent(None)
            self.selected_cross.deleteLater()
            self.selected_cross = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        ui_idl = self.window()
        ui_idl.add_all_crosses(event)

    def add_cross(self, event):
        cross = DraggableCross(self)
        cross.setGeometry(event.pos().x() - 10, event.pos().y() - 10, 20, 20)
        cross.load_png(CROSS_DIR)
        cross.show()


class UiIdl(UiReplay):
    # make this function public, CustomQLabel will use it
    def add_all_crosses(self, event):
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self._display_frame[i].add_cross(event)

    def __init__(self, debug_mode: bool):
        # pass debug_mode parameter to the parent class
        super().__init__(debug_mode)

    def _init_ui_names(self):
        # before _init_ui_names()
        self._display_frame_ct = CustomQLabel(self._central_widget)
        self._display_frame_pt = CustomQLabel(self._central_widget)
        self._display_frame_mrt1 = CustomQLabel(self._central_widget)
        self._display_frame_mrt2 = CustomQLabel(self._central_widget)

        super()._init_ui_names()

        self._text_label["annotation.tools"] = self._text_label_annotation_tools
        self._text_label["idl.gtvt.progress"] = self._text_label_idl_gtvt_progress

        self.__btn = Dict()
        self.__btn["pen"] = self._btn_pen
        self.__btn["eraser"] = self._btn_eraser
        self.__btn["clear"] = self._btn_clear
        self.__btn["confirm"] = self._btn_confirm

    def _init_member_var(self, debug_mode: bool):
        super()._init_member_var(debug_mode)

        self.__idl_step = CHOOSE_PATIENT

        # keep idl.gtvt/gtvn id unchanged
        cur_time = Time.get_cur_time_str()
        for i in ["gtvt", "gtvn"]:
            self._idl_id[i] = "idl.{}_".format(i) + cur_time
            if debug_mode:
                self._idl_id[i] += "_" + g.DELETE_FLAG

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F12:
            pass

        # delete selected cross
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            for i in ["ct", "pt", "mrt1", "mrt2"]:
                self._display_frame[i].delete_selected_cross()

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def __confirm_annotation(self):
        print("OK")

    def __clear_annotation(self):
        print("clear annotation")

    def __select_pen(self):
        print("select pen")

    def __select_eraser(self):
        print("select eraser")

    def _init_side_bar(self):
        super()._init_side_bar()

        # hide idl.gtvt/gtvn controls
        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._text_label[i].hide()
            self._combox[i].hide()
            self._arrow_btn["prev.{}".format(i)].hide()
            self._arrow_btn["next.{}".format(i)].hide()

        # show annotation controls
        self._text_box_annotation_msg.show()
        self._progress_bar_idl_gtvt.show()
        for i in ["annotation.tools", "idl.gtvt.progress"]:
            self._text_label[i].show()
        for i in ["pen", "eraser", "clear", "confirm"]:
            self.__btn[i].show()

        # set text
        self._text_box_annotation_msg.setText("Please Select a Patient")
        self._text_label["annotation.tools"].setText("Annotation Tools")
        self._text_label["idl.gtvt.progress"].setText("GTVt Retraining Progress")

        # set fonts
        for i in ["annotation.tools", "idl.gtvt.progress"]:
            self._text_label[i].setFont(self._font_bold)
        self._text_box_annotation_msg.setFont(self._font_bold)

        # set read only
        self._text_box_annotation_msg.setReadOnly(True)

        # connect ui to functions
        # (put this at the end, because these functions will need the initialization above)
        self.__btn["pen"].clicked.connect(self.__select_pen)
        self.__btn["eraser"].clicked.connect(self.__select_eraser)
        self.__btn["clear"].clicked.connect(self.__clear_annotation)
        self.__btn["confirm"].clicked.connect(self.__confirm_annotation)

    def _refresh_side_bar(self):
        left, top, width, gap, text_height, bar_height = super()._refresh_side_bar(
            widgets_to_display=["patient"]
        )

        annotation_msg_box_height = 80
        annotation_btn_width = 50

        # annotation message box
        top += gap
        rect = QRect(left, top, width, annotation_msg_box_height)
        self._text_box_annotation_msg.setGeometry(rect)
        top += annotation_msg_box_height

        # annotation tools
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label["annotation.tools"].setGeometry(rect)
        self._text_label["annotation.tools"].show()
        top += text_height
        tmp_left = left
        tmp_gap = round((width - 4 * annotation_btn_width) / 3)
        for i in ["pen", "eraser", "clear", "confirm"]:
            rect = QRect(tmp_left, top, annotation_btn_width, bar_height)
            self.__btn[i].setGeometry(rect)
            self.__btn[i].show()
            tmp_left += tmp_gap + annotation_btn_width
        top += bar_height

        # idl gtvt retraining progress bar
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label_idl_gtvt_progress.setGeometry(rect)
        self._text_label_idl_gtvt_progress.show()
        top += text_height
        rect = QRect(left, top, width, bar_height)
        self._progress_bar_idl_gtvt.setGeometry(rect)
        self._progress_bar_idl_gtvt.show()

    def _choose_baseline(self):
        # self._reset_zoomin()
        self._clear_img_data()
        self._clear_display_frames()

        self._baseline_id = "baseline_real.idl"
        # run these 2 lines after self._baseline_id is confirmed
        self._load_dataset_dir_and_nii_spacing()
        self._fill_combox_patient()
        self._combox["patient"].setCurrentIndex(-1)  # show nothing

        # run this after patient combox current text is set up
        self._enable_arrow_btns("patient")

        # create idl folders (after baseline_id is confirmed)
        for i in ["gtvt", "gtvn"]:
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
        self._cur_patient = self._combox["patient"].currentText()
        # self._reset_zoomin()

        # load multi-modal imgs only, no need to load labels
        self._load_multi_modal_imgs()

        # get slice id (after multi-modal imgs are loaded)
        self._cur_slice = self._get_middle_slice_id()

        self._choose_idl_gtvt()
        self._choose_idl_gtvn()
        self._refresh_imgs()
        self._refresh_title()

        # load idl step
        for i in ["gtvt", "gtvn"]:
            idl_step_json_path = os.path.join(
                g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i], "idl_step.json"
            )

            # if idl step json file does not exist, create it
            if not os.path.exists(idl_step_json_path):
                self.__idl_step = CLICK_GTVT_CENTER
                self.__update_annotation_msg()
                idl_step_dict = Dict()
                idl_step_dict["patient={}".format(self._cur_patient)] = self.__idl_step
                Json.save(idl_step_dict, idl_step_json_path)

            else:
                # load idl step from json file
                self.__idl_step = Json.load(idl_step_json_path)[
                    "patient={}".format(self._cur_patient)
                ]
                self.__update_annotation_msg()

    def __update_annotation_msg(self):
        if self.__idl_step == CLICK_GTVT_CENTER:
            self._text_box_annotation_msg.setText(
                "Please click the center of GTVt, then press OK"
            )

        elif self.__idl_step == DELINEATE_GTVT:
            self._text_box_annotation_msg.setText(
                "Please delineate the countour of GTVt on transvers/coronal/sagittal plane, then press OK"
            )

        elif self.__idl_step == CLICK_GTVN_CENTER:
            self._text_box_annotation_msg.setText(
                "Please click the center of GTVns, then press OK"
            )

        elif self.__idl_step == CORRECTION:
            self._text_box_annotation_msg.setText(
                "Please correct the predictions, then press OK"
            )

        else:
            Debug.error_exit("self.__idl_step value error")

    def _choose_idl_gtvt(self):
        self.__choose_idl(gtv="gtvt")

    def _choose_idl_gtvn(self):
        self.__choose_idl(gtv="gtvn")

    def __choose_idl(self, gtv: str):
        patient_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id[gtv],
            "patients",
            "patient={}".format(self._cur_patient),
        )

        # current patient dir exists
        if os.path.exists(patient_dir):
            # choose the last round
            pred_path = Directory.get_sub_folders(
                patient_dir, key_word="round=", full_path=True
            )[-1]
            pred_path = os.path.join(pred_path, "{}_pred.nii".format(gtv))

            # find idl pred, load it
            if os.path.exists(pred_path):
                self._3d_imgs["{}.pred".format(gtv)] = Img.binarize(
                    self._load_img(pred_path)
                )
            # cant find idl pred, clear 3d img
            else:
                self._3d_imgs["{}.pred".format(gtv)] = None

        # cant find cur patient dir
        else:
            self._3d_imgs["{}.pred".format(gtv)] = None
