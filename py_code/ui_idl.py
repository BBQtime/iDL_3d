import enum
import os
import platform
import sys
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Tuple

import cv2
import numpy as np
from custom import Dict, Explorer
from custom import Global as g
from custom import Img, Json, List, Nii, ValueUtils
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtWidgets import QApplication, QButtonGroup, QMainWindow, QRubberBand
from Ui_main_window import Ui_MainWindow


class Step(enum.Enum):
    CHOOSE_PATIENT = 0
    CLICK_GTVT_CENTER = 1
    ANNOTATE_GTVT = 2
    CLICK_GTVN_CENTER = 3
    IDL_COMPLETE = 4


class UiCore(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.__init_ui_names()
        self.__init_member_var()
        self.__init_zoomin()
        self.__init_color()
        self.__init_side_bar()  # after __init_member_var(), function connection needed

        self.__clear_img_data()
        self.__refresh_title()  # after __init_member_var()

        # resize
        self.resize(1200, 800)  # set origin size
        self.showMaximized()

        # load first baseline result
        self.__choose_baseline()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_F12:
            if self.__replay_mode:
                self.__replay_mode = False
                self.__annotation_step = Step.CHOOSE_PATIENT
                self.__text_box["annotation.msg"].setText("Please Select a Patient")
                best_baseline_id = "baseline_2023.02.27.07.08.09_3mm_best"
                baseline_id_list = self.__get_combox_contents(self.__combox["baseline"])
                if best_baseline_id in baseline_id_list:
                    self.__combox["baseline"].setCurrentText(best_baseline_id)
                else:
                    pass
            else:
                self.__replay_mode = True

            self.__choose_baseline()
            self.__refresh_side_bar()

    def __init_member_var(self):
        self.__patients = Dict()
        self.__patients["test.inter"] = Json.load(g.DATASET_SPLIT_JSON_PATH)[
            "test.inter"
        ]
        self.__patients["test.inter"] = List(
            self.__patients["test.inter"]
        )  # str to List
        self.__patients["test.inter"].sort()
        self.__patients["idl.gtvt"] = None
        self.__patients["idl.gtvn"] = None
        self.__patient = None
        self.__round = None
        self.__slice = None  # starts from 0
        self.__nii_spacing = None
        self.__dataset_dir = None
        self.__scores = Dict()
        self.__imgs = Dict()
        self.__resize_pos = Dict()
        self.__bright = Dict()
        self.__contrast = Dict()
        self.__bright_contrast_modality = "ct"
        self.__side_bar_width = 300
        self.__replay_mode = False  # replay mode or real-time mode
        self.__annotation_step = Step.CHOOSE_PATIENT

    def __clear_img_data(self):
        self.__patients["idl.gtvt"] = None
        self.__patients["idl.gtvn"] = None

        for i in ["gtvt", "gtvn"]:
            self.__scores[i]["dsc"] = None
            self.__scores[i]["msd"] = None
            self.__scores[i]["hd95"] = None

        for i in [
            "ct",
            "pt",
            "mrt1",
            "mrt2",
            "gtvt.label",
            "gtvn.label",
            "gtvt.pred",
            "gtvn.pred",
            "gtvn.clicks",
        ]:
            self.__imgs[i] = None

        # resize position of ct/pt/mrt1/mrt2
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__resize_pos[i] = None

        # set image plane
        for i in ["transverse", "coronal", "sagittal"]:
            if self.__radio_btn[i].isChecked():
                self.__img_plane = i

    def __init_zoomin(self):
        self.__zoomin = Dict()
        self.__zoomin["rubber.band"] = QRubberBand(QRubberBand.Rectangle, self)
        self.__reset_zoomin()

    def __reset_zoomin(self):
        self.__zoomin["rubber.band"].hide()
        self.__zoomin["img"] = None
        self.__zoomin["start"] = None
        self.__zoomin["end"] = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        # loop 4 img frames
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            left = self.__display_frame[i].x()
            top = self.__display_frame[i].y()
            width = self.__display_frame[i].width()
            height = self.__display_frame[i].height()
            # if start pos is in current img frame
            if (
                event.x() >= left
                and event.x() <= left + width
                and event.y() >= top
                and event.y() <= top + height
            ):
                # already zoomed in, clear zoomin (only click in img frame area)
                if self.__zoomin["start"] is not None:
                    self.__reset_zoomin()
                    self.__refresh_imgs()
                    return
                # zoom in
                else:
                    self.__zoomin["img"] = i
                    self.__zoomin["start"] = event.pos()
                    rect = QRect(event.pos(), event.pos())
                    self.__zoomin["rubber.band"].setGeometry(rect.normalized())
                    self.__zoomin["rubber.band"].show()
                    return

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.__zoomin["start"] is None:  # or self.__zoomin["rubber.band"] is None:
            return
        self.__mouse_move_event(event)

    def __mouse_move_event(self, event):
        # limit zoomin frame in img frame
        display_frame = self.__display_frame[self.__zoomin["img"]]
        display_frame_right = display_frame.x() + display_frame.width() - 1
        if event.x() < display_frame.x():
            event_x = display_frame.x()
        elif event.x() > display_frame_right:
            event_x = display_frame_right
        else:
            event_x = event.x()
        display_frame_buttom = display_frame.y() + display_frame.height() - 1
        if event.y() < display_frame.y():
            event_y = display_frame.y()
        elif event.y() > display_frame_buttom:
            event_y = display_frame_buttom
        else:
            event_y = event.y()
        # resize zoomin frame
        self.__zoomin["end"] = QPoint(event_x, event_y)
        rect = QRect(
            self.__zoomin["start"],
            self.__zoomin["end"],
        )
        self.__zoomin["rubber.band"].setGeometry(rect.normalized())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        # not zoomed in
        if self.__zoomin["start"] is None:  # or self.__zoomin["rubber.band"] is None:
            return
        self.__mouse_move_event(event)
        self.__zoomin["rubber.band"].hide()
        # self.__zoomin["rubber.band"] = None

        # no data loaded
        if self.__imgs["pt"] is None:
            self.__reset_zoomin()
            return

        # zoomin size == 0
        if (
            abs(self.__zoomin["start"].x() - self.__zoomin["end"].x()) <= 1
            or abs(self.__zoomin["start"].y() - self.__zoomin["end"].y()) <= 1
        ):
            # print("zoomin size 0")
            self.__reset_zoomin()
            return
        self.__get_img_roi()

    def __get_img_roi(self):
        # make sure start point always < end point
        start_x = self.__zoomin["start"].x()
        start_y = self.__zoomin["start"].y()
        end_x = self.__zoomin["end"].x()
        end_y = self.__zoomin["end"].y()
        if start_x > end_x:
            x = start_x
            start_x = end_x
            end_x = x
        if start_y > end_y:
            y = start_y
            start_y = end_y
            end_y = y
        # get display_frame related position
        display_frame_left = self.__display_frame[self.__zoomin["img"]].x()
        display_frame_top = self.__display_frame[self.__zoomin["img"]].y()
        start_x -= display_frame_left
        end_x -= display_frame_left
        start_y -= display_frame_top
        end_y -= display_frame_top

        # get actual_img_area related position
        resize_pos = self.__resize_pos[self.__zoomin["img"]]
        start_x -= resize_pos["x"]
        start_y -= resize_pos["y"]
        end_x -= resize_pos["x"]
        end_y -= resize_pos["y"]
        # out of range
        if (start_x < 0 and end_x < 0) or (
            start_x > resize_pos["width"] and end_x > resize_pos["width"]
        ):
            self.__reset_zoomin()
            return
        if (start_y < 0 and end_y < 0) or (
            start_y > resize_pos["height"] and end_y > resize_pos["height"]
        ):
            self.__reset_zoomin()
            return
        # limit zoomin frame in image area
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > resize_pos["width"]:
            end_x = resize_pos["width"]
        if end_y > resize_pos["height"]:
            end_y = resize_pos["height"]

        # get actual zoom position
        if self.__img_plane == "sagittal":
            origin_width = self.__imgs["pt"].shape[1]
            origin_height = self.__imgs["pt"].shape[0]
            origin_height = round(
                origin_height * self.__nii_spacing[2] / self.__nii_spacing[1]
            )
        elif self.__img_plane == "coronal":
            origin_width = self.__imgs["pt"].shape[2]
            origin_height = self.__imgs["pt"].shape[0]
            origin_height = round(
                origin_height * self.__nii_spacing[2] / self.__nii_spacing[0]
            )
        else:
            origin_width = self.__imgs["pt"].shape[2]
            origin_height = self.__imgs["pt"].shape[1]

        start_x = round(start_x * origin_width / resize_pos["width"])
        end_x = round(end_x * origin_width / resize_pos["width"])
        start_y = round(start_y * origin_height / resize_pos["height"])
        end_y = round(end_y * origin_height / resize_pos["height"])

        self.__zoomin["start"] = QPoint(start_x, start_y)
        self.__zoomin["end"] = QPoint(end_x, end_y)
        self.__refresh_imgs()

    def __fit_display_frame(self, img, display_frame: QtWidgets.QLabel):
        err_msg = "MainWindow.__fit_display_frame(), img.shape should == 2 or 3"

        # image spacing resize
        if self.__img_plane == "sagittal":
            spacing_height = round(
                img.shape[0] * self.__nii_spacing[2] / self.__nii_spacing[1]
            )
            img = cv2.resize(
                img,
                (
                    img.shape[1],
                    spacing_height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        elif self.__img_plane == "coronal":
            spacing_height = round(
                img.shape[0] * self.__nii_spacing[2] / self.__nii_spacing[0]
            )
            img = cv2.resize(
                img,
                (
                    img.shape[1],
                    spacing_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        # zoom in
        if self.__zoomin["start"] is not None and self.__zoomin["end"] is not None:
            if len(img.shape) == 3:
                img = img[
                    self.__zoomin["start"].y() : self.__zoomin["end"].y(),
                    self.__zoomin["start"].x() : self.__zoomin["end"].x(),
                    :,
                ]
            elif len(img.shape) == 2:
                img = img[
                    self.__zoomin["start"].y() : self.__zoomin["end"].y(),
                    self.__zoomin["start"].x() : self.__zoomin["end"].x(),
                ]
            else:
                raise ValueError(err_msg)

        # resize to fit image frame
        origin_height = img.shape[0]
        origin_width = img.shape[1]
        resize_pos = Dict()
        resize_pos["x"], resize_pos["y"] = None, None
        resize_pos["width"], resize_pos["height"] = None, None
        final_width = display_frame.width()
        final_height = display_frame.height()

        # border on left and right side
        if origin_height * final_width > final_height * origin_width:
            resize_pos["width"] = int(final_height * origin_width / origin_height)
            resize_pos["height"] = final_height
            resize_pos["x"] = int((final_width - resize_pos["width"]) / 2)
            if resize_pos["x"] < 0:
                resize_pos["x"] = 0
            resize_pos["y"] = 0
            if len(img.shape) == 3:
                black_border = np.zeros((final_height, resize_pos["x"], 3), np.uint8)
            elif len(img.shape) == 2:
                black_border = np.zeros((final_height, resize_pos["x"]), np.uint8)
            else:
                raise ValueError(err_msg)
            img = cv2.resize(
                img,
                (resize_pos["width"], resize_pos["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=1)

        # border on up and down side
        else:
            resize_pos["width"] = final_width
            resize_pos["height"] = int(final_width * origin_height / origin_width)
            resize_pos["y"] = int((final_height - resize_pos["height"]) / 2)
            if resize_pos["y"] < 0:
                resize_pos["y"] = 0
            resize_pos["x"] = 0
            if len(img.shape) == 3:
                black_border = np.zeros((resize_pos["y"], final_width, 3), np.uint8)
            elif len(img.shape) == 2:
                black_border = np.zeros((resize_pos["y"], final_width), np.uint8)
            else:
                raise ValueError(err_msg)
            img = cv2.resize(
                img,
                (resize_pos["width"], resize_pos["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=0)

        # smooth img
        return img, resize_pos

    def __init_color(self):
        self.__color = Dict()
        self.__color["gtvt.label"] = (0, 255, 255)  # light blue
        self.__color["gtvn.label"] = (0, 150, 255)  # dark blue
        self.__color["gtvt.pred"] = (255, 255, 0)  # yellow
        self.__color["gtvn.pred"] = (255, 128, 0)  # orange
        self.__color["gtvt.annotation"] = (0, 255, 64)  # green
        self.__color["gtvn.annotation"] = (255, 70, 200)  # pink
        self.__color["score.text"] = self.__color["gtvt.annotation"]

    def __init_ui_names(self):
        self.__display_frame = Dict()
        self.__display_frame["ct"] = self._display_frame_ct
        self.__display_frame["pt"] = self._display_frame_pt
        self.__display_frame["mrt1"] = self._display_frame_mrt1
        self.__display_frame["mrt2"] = self._display_frame_mrt2

        self.__text_label = Dict()
        self.__text_label["baseline"] = self._text_label_baseline
        self.__text_label["idl.gtvs"] = self._text_label_idl_gtvs
        self.__text_label["idl.gtvt"] = self._text_label_idl_gtvt
        self.__text_label["idl.gtvn"] = self._text_label_idl_gtvn
        self.__text_label["patient"] = self._text_label_patient
        self.__text_label["round"] = self._text_label_round
        self.__text_label["bright"] = self._text_label_bright
        self.__text_label["contrast"] = self._text_label_contrast
        self.__text_label["zoom"] = self._text_label_zoom
        self.__text_label["annotation.tools"] = self._text_label_annotation_tools
        self.__text_label["idl.gtvt.progress"] = self._text_label_idl_gtvt_progress

        self.__text_box = Dict()
        self.__text_box["annotation.msg"] = self._text_box_annotation_msg

        self.__combox = Dict()
        self.__combox["baseline"] = self._combox_baseline
        self.__combox["idl.gtvs"] = self._combox_idl_gtvs
        self.__combox["idl.gtvt"] = self._combox_idl_gtvt
        self.__combox["idl.gtvn"] = self._combox_idl_gtvn
        self.__combox["patient"] = self._combox_patient
        self.__combox["round"] = self._combox_round

        self.__btn = Dict()
        self.__btn["pen"] = self._btn_pen
        self.__btn["eraser"] = self._btn_eraser
        self.__btn["clear"] = self._btn_clear
        self.__btn["confirm"] = self._btn_confirm

        self.__arrow_btn = Dict()
        self.__arrow_btn["prev.baseline"] = self._btn_prev_baseline
        self.__arrow_btn["next.baseline"] = self._btn_next_baseline
        self.__arrow_btn["prev.idl.gtvs"] = self._btn_prev_idl_gtvs
        self.__arrow_btn["next.idl.gtvs"] = self._btn_next_idl_gtvs
        self.__arrow_btn["prev.idl.gtvt"] = self._btn_prev_idl_gtvt
        self.__arrow_btn["next.idl.gtvt"] = self._btn_next_idl_gtvt
        self.__arrow_btn["prev.idl.gtvn"] = self._btn_prev_idl_gtvn
        self.__arrow_btn["next.idl.gtvn"] = self._btn_next_idl_gtvn
        self.__arrow_btn["prev.patient"] = self._btn_prev_patient
        self.__arrow_btn["next.patient"] = self._btn_next_patient
        self.__arrow_btn["prev.round"] = self._btn_prev_round
        self.__arrow_btn["next.round"] = self._btn_next_round

        self.__radio_btn = Dict()
        self.__radio_btn["ct"] = self._radio_btn_ct
        self.__radio_btn["pt"] = self._radio_btn_pt
        self.__radio_btn["mrt1"] = self._radio_btn_mrt1
        self.__radio_btn["mrt2"] = self._radio_btn_mrt2
        self.__radio_btn["transverse"] = self._radio_btn_transverse
        self.__radio_btn["coronal"] = self._radio_btn_coronal
        self.__radio_btn["sagittal"] = self._radio_btn_sagittal

        self.__slider = Dict()
        self.__slider["bright"] = self._slider_bright
        self.__slider["contrast"] = self._slider_contrast
        self.__slider["zoom"] = self._slider_zoom

        # set label background black
        pal = QPalette()
        pal.setColor(QPalette.Window, Qt.black)
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__display_frame[i].setObjectName("")
            self.__display_frame[i].setAutoFillBackground(True)
            self.__display_frame[i].setPalette(pal)

    def __init_side_bar(self):
        # set text
        self.__text_label["baseline"].setText("Choose Baseline Result")
        self.__text_label["idl.gtvs"].setText("Choose iDL GTVs Result")
        self.__text_label["idl.gtvt"].setText("Choose iDL GTVt Result")
        self.__text_label["idl.gtvn"].setText("Choose iDL GTVn Result")
        self.__text_label["patient"].setText("Choose Patient")
        self.__text_label["round"].setText("Choose Update Round")
        self.__text_label["bright"].setText("Brightness (CT)")
        self.__text_label["contrast"].setText("Contrast (CT)")
        self.__text_label["zoom"].setText("Zoom In")
        self.__text_label["annotation.tools"].setText("Annotation Tools")
        self.__text_label["idl.gtvt.progress"].setText("GTVt Retraining Progress")
        self.__text_box["annotation.msg"].setText("Please Select a Patient")
        self.__radio_btn["ct"].setText("CT")
        self.__radio_btn["pt"].setText("PT")
        self.__radio_btn["mrt1"].setText("MR-T1")
        self.__radio_btn["mrt2"].setText("MR-T2")
        self.__radio_btn["transverse"].setText("Transverse")
        self.__radio_btn["coronal"].setText("Coronal")
        self.__radio_btn["sagittal"].setText("Sagittal")

        # set font
        font = self.__text_label["baseline"].font()
        font.setPointSize(8)
        font.setBold(True)
        # set font of text labels
        for i in [
            "baseline",
            "idl.gtvs",
            "idl.gtvt",
            "idl.gtvn",
            "patient",
            "round",
            "bright",
            "contrast",
            "zoom",
            "annotation.tools",
            "idl.gtvt.progress",
        ]:
            self.__text_label[i].setFont(font)
        # set font of text boxes
        for i in ["annotation.msg"]:
            self.__text_box[i].setFont(font)
        # set font of radio buttons
        for i in ["transverse", "coronal", "sagittal", "ct", "pt", "mrt1", "mrt2"]:
            self.__radio_btn[i].setFont(font)
        # set font of comboboxes
        font.setBold(False)
        for i in ["baseline", "idl.gtvs", "idl.gtvt", "idl.gtvn", "patient", "round"]:
            self.__combox[i].setFont(font)

        # set combobox dropdown width: 700px
        for i in ["baseline", "idl.gtvs", "idl.gtvt", "idl.gtvn"]:
            self.__combox[i].setStyleSheet(
                """*
                QComboBox QAbstractItemView
                {
                    min-width: 500px;
                }
                """
            )

        # fill the baseline combobox, format "baseline_id"
        self.__combox["baseline"].addItems(
            Explorer.get_sub_folders(
                g.TRAIN_RESULTS_DIR, key_word="baseline_", shuffle=False
            )
        )

        # set initial state
        for i in ["baseline", "idl.gtvs", "idl.gtvt", "idl.gtvn", "patient", "round"]:
            self.__arrow_btn["prev.{}".format(i)].setArrowType(Qt.LeftArrow)
            self.__arrow_btn["next.{}".format(i)].setArrowType(Qt.RightArrow)

        for i in ["idl.gtvs", "idl.gtvt", "idl.gtvn", "patient", "round"]:
            self.__combox[i].setEnabled(False)
            self.__arrow_btn["prev.{}".format(i)].setEnabled(False)
            self.__arrow_btn["next.{}".format(i)].setEnabled(False)

        self.__text_box["annotation.msg"].setReadOnly(True)

        # Add radio buttons to the button group
        self.__btn_group_bright_contrast = QButtonGroup()
        self.__btn_group_bright_contrast.addButton(self.__radio_btn["ct"])
        self.__btn_group_bright_contrast.addButton(self.__radio_btn["pt"])
        self.__btn_group_bright_contrast.addButton(self.__radio_btn["mrt1"])
        self.__btn_group_bright_contrast.addButton(self.__radio_btn["mrt2"])
        self.__btn_group_plane = QButtonGroup()
        self.__btn_group_plane.addButton(self.__radio_btn["transverse"])
        self.__btn_group_plane.addButton(self.__radio_btn["coronal"])
        self.__btn_group_plane.addButton(self.__radio_btn["sagittal"])

        # radio btns checked or not
        self.__radio_btn["ct"].setChecked(True)
        self.__radio_btn["pt"].setChecked(False)
        self.__radio_btn["mrt1"].setChecked(False)
        self.__radio_btn["mrt2"].setChecked(False)
        self.__radio_btn["transverse"].setChecked(True)
        self.__radio_btn["coronal"].setChecked(False)
        self.__radio_btn["sagittal"].setChecked(False)

        # set slider range and default value
        self.__slider["bright"].setMinimum(-128)
        self.__slider["bright"].setMaximum(128)
        self.__slider["bright"].setValue(0)
        self.__slider["contrast"].setMinimum(0)
        self.__slider["contrast"].setMaximum(200)
        self.__slider["contrast"].setValue(100)
        self.__slider["zoom"].setMinimum(100)
        self.__slider["zoom"].setMaximum(200)
        self.__slider["zoom"].setValue(100)
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__bright[i] = self.__slider["bright"].value()
            self.__contrast[i] = self.__slider["contrast"].value()

        # connect ui to functions
        # (put this at the end, because these functions will need the initialization above)
        # e.g. __set_bright_contrast_modality will need value of self.__bright and self.__contrast
        self.__combox["baseline"].activated.connect(self.__choose_baseline)
        self.__arrow_btn["prev.baseline"].clicked.connect(self.__choose_prev_baseline)
        self.__arrow_btn["next.baseline"].clicked.connect(self.__choose_next_baseline)

        self.__combox["idl.gtvs"].activated.connect(self.__choose_idl_gtvs)
        self.__arrow_btn["prev.idl.gtvs"].clicked.connect(self.__choose_prev_idl_gtvs)
        self.__arrow_btn["next.idl.gtvs"].clicked.connect(self.__choose_next_idl_gtvs)

        self.__combox["idl.gtvt"].activated.connect(self.__choose_idl_gtvt)
        self.__arrow_btn["prev.idl.gtvt"].clicked.connect(self.__choose_prev_idl_gtvt)
        self.__arrow_btn["next.idl.gtvt"].clicked.connect(self.__choose_next_idl_gtvt)

        self.__combox["idl.gtvn"].activated.connect(self.__choose_idl_gtvn)
        self.__arrow_btn["prev.idl.gtvn"].clicked.connect(self.__choose_prev_idl_gtvn)
        self.__arrow_btn["next.idl.gtvn"].clicked.connect(self.__choose_next_idl_gtvn)

        self.__combox["patient"].activated.connect(self.__choose_patient)
        self.__arrow_btn["prev.patient"].clicked.connect(self.__choose_prev_patient)
        self.__arrow_btn["next.patient"].clicked.connect(self.__choose_next_patient)

        self.__combox["round"].activated.connect(self.__choose_round)
        self.__arrow_btn["prev.round"].clicked.connect(self.__choose_prev_round)
        self.__arrow_btn["next.round"].clicked.connect(self.__choose_next_round)

        self.__btn["pen"].clicked.connect(self.__select_pen)
        self.__btn["eraser"].clicked.connect(self.__select_eraser)
        self.__btn["clear"].clicked.connect(self.__clear_annotation)
        self.__btn["confirm"].clicked.connect(self.__confirm_annotation)

        for i in ["bright", "contrast"]:
            self.__slider[i].valueChanged.connect(self.__refresh_imgs)

        # for i in ["transverse", "coronal", "sagittal"]:
        #     self.__radio_btn[i].toggled.connect(self.__set_img_plane)
        self.__btn_group_plane.buttonClicked.connect(self.__set_img_plane)

        # for i in ["ct", "pt", "mrt1", "mrt2"]:
        #     self.__radio_btn[i].toggled.connect(self.__set_bright_contrast_modality)
        self.__btn_group_bright_contrast.buttonClicked.connect(
            self.__set_bright_contrast_modality
        )

    def __confirm_annotation(self):
        if self.__annotation_step == Step.CLICK_GTVT_CENTER:
            self.__text_box["annotation.msg"].setText(
                "Please delineate the GTVt on transvers/coronal/sagittal plane, then press the 'Confirm' button (green button)"
            )
            self.__annotation_step = Step.ANNOTATE_GTVT

        elif self.__annotation_step == Step.ANNOTATE_GTVT:
            self.__text_box["annotation.msg"].setText(
                "Please click the center of GTVns, then press the 'Confirm' button (green button)"
            )
            # gtvt training process
            ###################################
            # self.__combox["idl.gtvt"].addItem(new_gtvt_id)
            # self.__combox["idl.gtvt"].setCurrentText(new_gtvt_id)
            self.__annotation_step = Step.CLICK_GTVN_CENTER

        elif self.__annotation_step == Step.CLICK_GTVN_CENTER:
            # gtvn training process
            ###################################
            # self.__clear_img_data()
            # reset ui
            # for i in ["idl.gtvs", "idl.gtvt", "idl.gtvn", "patient", "round"]:
            #     self.__combox[i].clear()
            #     self.__combox[i].setEnabled(False)
            #     self.__arrow_btn["prev.{}".format(i)].setEnabled(False)
            #     self.__arrow_btn["next.{}".format(i)].setEnabled(False)

            self.__combox["idl.gtvn"].setCurrentText(
                "idl.gtvn_2023.05.26.12.06.15_best"
            )
            self.__choose_patient(idx=None, reset_patient=False)
            self.__annotation_step = Step.IDL_COMPLETE
        else:
            pass

    def __clear_annotation(self):
        print("clear annotation")

    def __select_pen(self):
        print("select pen")

    def __select_eraser(self):
        print("select eraser")

    def __refresh_side_bar(self):
        # these are adjustable
        left = 30
        top = 0
        text_height = 25
        bar_height = 25
        slider_height = 20
        annotation_msg_box_height = 120
        arrow_btn_width = 30
        annotation_btn_width = 50

        if platform.system().lower() == "linux":
            gap = 27
        else:  # windows
            gap = 50

        radio_btn_height = 25
        radio_btn_width = Dict()
        radio_btn_width["ct"] = radio_btn_width["pt"] = 45
        radio_btn_width["mrt1"] = radio_btn_width["mrt2"] = 60
        radio_btn_width["transverse"] = 90
        radio_btn_width["coronal"] = 70
        radio_btn_width["sagittal"] = 70
        radio_btn_gap = Dict()
        radio_btn_gap["bright.contrast"] = 10
        radio_btn_gap["planes"] = 6

        # side bar location
        side_bar_x = self.geometry().width() - self.__side_bar_width
        width = self.__side_bar_width - left * 2
        left += side_bar_x

        # hide and show text label / comboxes / btns
        if self.__replay_mode:
            for i in ["baseline", "idl.gtvt", "idl.gtvn", "round"]:
                self.__text_label[i].show()
                self.__arrow_btn["prev.{}".format(i)].show()
                self.__combox[i].show()
                self.__arrow_btn["next.{}".format(i)].show()
        else:
            for i in ["baseline", "idl.gtvs", "idl.gtvt", "idl.gtvn", "round"]:
                self.__text_label[i].hide()
                self.__arrow_btn["prev.{}".format(i)].hide()
                self.__combox[i].hide()
                self.__arrow_btn["next.{}".format(i)].hide()

        # set position of text label / comboxes / btns
        if self.__replay_mode:
            ui_name_list = [
                "baseline",
                "idl.gtvt",
                "idl.gtvn",
                "patient",
                "round",
            ]
        else:
            ui_name_list = ["patient"]
        for i in ui_name_list:
            # text label
            top += gap
            rect = QRect(left, top, width, text_height)
            self.__text_label[i].setGeometry(rect)
            top += text_height

            # btn prev
            tmp_left = left
            rect = QRect(tmp_left, top, arrow_btn_width, bar_height)
            self.__arrow_btn["prev.{}".format(i)].setGeometry(rect)

            # combobox
            tmp_left += arrow_btn_width
            rect = QRect(tmp_left + 1, top, width - arrow_btn_width * 2 - 2, bar_height)
            self.__combox[i].setGeometry(rect)

            # btn next
            tmp_left += width - arrow_btn_width * 2
            rect = QRect(tmp_left, top, arrow_btn_width, bar_height)
            self.__arrow_btn["next.{}".format(i)].setGeometry(rect)

            # next element
            top += bar_height

        # brightness and contrast radio btns
        top += gap
        tmp_left = left
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            rect = QRect(tmp_left, top, radio_btn_width[i], radio_btn_height)
            self.__radio_btn[i].setGeometry(rect)
            tmp_left += radio_btn_gap["bright.contrast"] + radio_btn_width[i]
        top += radio_btn_height

        # brightness and contrast bars
        for i in ["bright", "contrast"]:
            rect = QRect(left, top, width, text_height)
            self.__text_label[i].setGeometry(rect)
            top += text_height
            rect = QRect(left, top, width, slider_height)
            self.__slider[i].setGeometry(rect)
            top += slider_height

        # img plane
        top += gap
        tmp_left = left
        for i in ["transverse", "coronal", "sagittal"]:
            rect = QRect(tmp_left, top, radio_btn_width[i], radio_btn_height)
            self.__radio_btn[i].setGeometry(rect)
            tmp_left += radio_btn_gap["planes"] + radio_btn_width[i]
        top += radio_btn_height

        # zoom
        top += gap
        rect = QRect(left, top, width, text_height)
        self.__text_label["zoom"].setGeometry(rect)
        top += text_height
        rect = QRect(left, top, width, slider_height)
        self.__slider["zoom"].setGeometry(rect)
        top += slider_height

        # annotation message box
        if self.__replay_mode:
            self.__text_box["annotation.msg"].hide()
        else:
            top += gap
            rect = QRect(left, top, width, annotation_msg_box_height)
            self.__text_box["annotation.msg"].setGeometry(rect)
            self.__text_box["annotation.msg"].show()
            top += annotation_msg_box_height

        # annotation tools
        if self.__replay_mode:
            self.__text_label["annotation.tools"].hide()
            for i in ["pen", "eraser", "clear", "confirm"]:
                self.__btn[i].hide()
        else:
            top += gap
            rect = QRect(left, top, width, text_height)
            self.__text_label["annotation.tools"].setGeometry(rect)
            self.__text_label["annotation.tools"].show()
            top += text_height
            tmp_left = left
            tmp_gap = round((width - 4 * annotation_btn_width) / 3)
            for i in ["pen", "eraser", "clear", "confirm"]:
                rect = QRect(tmp_left, top, annotation_btn_width, bar_height)
                self.__btn[i].setGeometry(rect)
                self.__btn[i].show()
                tmp_left += tmp_gap + annotation_btn_width
            top += bar_height

        # idl gtvt retraining progress
        if self.__replay_mode:
            self._text_label_idl_gtvt_progress.hide()
            self._progress_bar_idl_gtvt.hide()
        else:
            top += gap
            rect = QRect(left, top, width, text_height)
            self._text_label_idl_gtvt_progress.setGeometry(rect)
            self._text_label_idl_gtvt_progress.show()
            top += text_height
            rect = QRect(left, top, width, bar_height)
            self._progress_bar_idl_gtvt.setGeometry(rect)
            self._progress_bar_idl_gtvt.show()

    def __load_img(self, path: str):
        img = Nii.load(path, binary=False)
        # ct windowing before normalization
        if "CT" in path:
            img = Img.ct_windowing(img)
        img = Img.normalize(img)
        # turn upside down
        img = np.flip(m=img, axis=0)
        # img = np.rot90(m=img, k=2, axes=(0, 1))
        return img

    def __set_img_plane(self):
        for i in ["transverse", "coronal", "sagittal"]:
            if self.__radio_btn[i].isChecked():
                self.__img_plane = i
                break

        # update and check slice_id (starts from 0)
        slices_count = self.__get_slices_count()
        if slices_count > 0:
            self.__slice = round(slices_count / 2) - 1
            self.__slice = ValueUtils.limit_range(self.__slice, (0, slices_count - 1))
        self.__reset_zoomin()
        self.__refresh_imgs()
        self.__refresh_title()

    def __set_bright_contrast_modality(self):
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            if self.__radio_btn[i].isChecked():
                self.__bright_contrast_modality = i
                break

        if self.__bright_contrast_modality == "ct":
            key_word = "CT"
        if self.__bright_contrast_modality == "pt":
            key_word = "PT"
        if self.__bright_contrast_modality == "mrt1":
            key_word = "MR-T1"
        if self.__bright_contrast_modality == "mrt2":
            key_word = "MR-T2"

        # switch brightness and contrast value (to new modality)
        self.__slider["bright"].setValue(self.__bright[self.__bright_contrast_modality])
        self.__slider["contrast"].setValue(
            self.__contrast[self.__bright_contrast_modality]
        )

        self.__text_label["bright"].setText("Brightness ({})".format(key_word))
        self.__text_label["contrast"].setText("Contrast ({})".format(key_word))

    def __clear_display_frames(self):
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            width = self.__display_frame[i].width()
            height = self.__display_frame[i].height()
            black_img = np.zeros([width, height, 3])
            qt_image = QImage(
                black_img,
                self.__display_frame[i].width(),
                self.__display_frame[i].height(),
                self.__display_frame[i].width() * 3,
                QImage.Format_RGB888,
            )
            self.__display_frame[i].setPixmap(QPixmap.fromImage(qt_image))

    def __enable_arrow_btns(self, combox_name: str):
        # enable/disable prev/next round buttons
        idx = self.__combox[combox_name].currentIndex()
        if idx == 0:
            self.__arrow_btn["prev.{}".format(combox_name)].setEnabled(False)
        else:
            self.__arrow_btn["prev.{}".format(combox_name)].setEnabled(True)

        if idx == (self.__combox[combox_name].count() - 1):
            self.__arrow_btn["next.{}".format(combox_name)].setEnabled(False)
        else:
            self.__arrow_btn["next.{}".format(combox_name)].setEnabled(True)

    def __get_combox_contents(self, combox: QtWidgets.QComboBox):
        content_list = List()
        for i in range(combox.count()):
            content_list.append(combox.itemText(i))
        return content_list

    def __choose_baseline(self):
        self.__reset_zoomin()
        self.__clear_img_data()
        self.__clear_display_frames()

        # run this after baseline combox current text is decided
        if self.__replay_mode:
            self.__enable_arrow_btns("baseline")
        else:
            self.__arrow_btn["prev.baseline"].setEnabled(False)
            self.__arrow_btn["next.baseline"].setEnabled(False)

        # reset ui
        for i in ["idl.gtvs", "idl.gtvt", "idl.gtvn", "patient", "round"]:
            self.__combox[i].clear()
            self.__combox[i].setEnabled(False)
            self.__arrow_btn["prev.{}".format(i)].setEnabled(False)
            self.__arrow_btn["next.{}".format(i)].setEnabled(False)

        baseline_id = self.__combox["baseline"].currentText()

        # fill idl.gtvt/gtvn/gtvs combobox
        for gtv in ["idl.gtvt", "idl.gtvn", "idl.gtvs"]:
            train_id_list = ["baseline"]
            train_id_list += Explorer.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_DIR, baseline_id),
                key_word=gtv,
                full_path=False,
            )
            self.__combox[gtv].addItems(train_id_list)

            # set current idl_gtvt_id
            if self.__replay_mode:
                self.__combox[gtv].setEnabled(True)
                # no idl found, show baseline
                if len(train_id_list) == 1:
                    self.__combox[gtv].setCurrentText(train_id_list[0])
                # first idl_gtvt_id
                else:
                    self.__combox[gtv].setCurrentText(train_id_list[1])
            else:
                # "baseline"
                self.__combox[gtv].setCurrentText("baseline")

            if gtv == "idl.gtvt":
                self.__choose_idl_gtvt()
            elif gtv == "idl.gtvn":
                self.__choose_idl_gtvn()
            else:
                self.__choose_idl_gtvs()

    def __choose_idl_gtvs(self):
        return

    def __choose_idl_gtvn(self):
        # run this after idl gtvn combox is filled
        if self.__replay_mode:
            self.__enable_arrow_btns("idl.gtvn")

        # reset ui
        for i in ["patient", "round"]:
            self.__combox[i].clear()
            self.__combox[i].setEnabled(False)
            self.__arrow_btn["prev.{}".format(i)].setEnabled(False)
            self.__arrow_btn["next.{}".format(i)].setEnabled(False)

        # confirm idl gtvn patients
        baseline_id = self.__combox["baseline"].currentText()
        if self.__combox["idl.gtvn"].currentText() == "baseline":
            self.__patients["idl.gtvn"] = Explorer.get_sub_folders(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    "baseline",
                    "cross_valid",
                    "patients",
                )
            )
        else:
            idl_gtvn_id = self.__combox["idl.gtvn"].currentText()
            self.__patients["idl.gtvn"] = Explorer.get_sub_folders(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    idl_gtvn_id,
                    "cross_valid",
                    "patients",
                )
            )
        # from "patient=123" to "123"
        for i in range(len(self.__patients["idl.gtvn"])):
            self.__patients["idl.gtvn"][i] = self.__patients["idl.gtvn"][i][
                len("patient=") :
            ]
        # idl gtvn patients & testset patients
        self.__patients["idl.gtvn"].find_identical_items(self.__patients["test.inter"])

        # fill combobox
        if self.__patients["idl.gtvt"] is not None:
            combox_patients = self.__patients["idl.gtvn"].copy()
            combox_patients.find_identical_items(self.__patients["idl.gtvt"])
            self.__combox["patient"].addItems(combox_patients)
            self.__combox["patient"].setEnabled(True)

            # choose patient automatically if replay mode
            if self.__replay_mode:
                # try not to reset patient when idl_gtvt_id is changed
                if self.__patient not in combox_patients:
                    reset_patient = True
                else:
                    reset_patient = False
                self.__choose_patient(idx=None, reset_patient=reset_patient)
            else:
                self.__patient = None
                self.__combox["patient"].setCurrentIndex(-1)

    def __choose_idl_gtvt(self):
        # run this after idl gtvt combox is filled
        if self.__replay_mode:
            self.__enable_arrow_btns("idl.gtvt")

        # reset ui
        for i in ["patient", "round"]:
            self.__combox[i].clear()
            self.__combox[i].setEnabled(False)
            self.__arrow_btn["prev.{}".format(i)].setEnabled(False)
            self.__arrow_btn["next.{}".format(i)].setEnabled(False)

        baseline_id = self.__combox["baseline"].currentText()
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, "baseline")

        # load slice thickness from baseline hyper
        fold_dirs = Explorer.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        )
        slice_thick = Json.load(os.path.join(fold_dirs[0], "hyper.json"))["slice.thick"]
        self.__nii_spacing = g.NII_SPACING[slice_thick]
        self.__dataset_dir = g.DATASET_DIR[slice_thick]

        # confirm idl gtvt patients
        if self.__combox["idl.gtvt"].currentText() == "baseline":
            self.__patients["idl.gtvt"] = Explorer.get_sub_folders(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    "baseline",
                    "cross_valid",
                    "patients",
                )
            )
        else:
            idl_gtvt_id = self.__combox["idl.gtvt"].currentText()
            self.__patients["idl.gtvt"] = Explorer.get_sub_folders(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    idl_gtvt_id,
                    "patients",
                )
            )
        # from "patient=123" to "123"
        for i in range(len(self.__patients["idl.gtvt"])):
            self.__patients["idl.gtvt"][i] = self.__patients["idl.gtvt"][i][
                len("patient=") :
            ]
        # idl gtvt patients & testset patients
        self.__patients["idl.gtvt"].find_identical_items(self.__patients["test.inter"])

        # fill combobox
        if self.__patients["idl.gtvn"] is not None:
            combox_patients = self.__patients["idl.gtvt"].copy()
            combox_patients.find_identical_items(self.__patients["idl.gtvn"])
            self.__combox["patient"].addItems(combox_patients)
            self.__combox["patient"].setEnabled(True)

            # choose patient automatically if replay mode
            if self.__replay_mode:
                # try not to reset patient when idl_gtvt_id is changed
                if self.__patient not in combox_patients:
                    reset_patient = True
                else:
                    reset_patient = False
                self.__choose_patient(idx=None, reset_patient=reset_patient)
            else:
                self.__patient = None
                self.__combox["patient"].setCurrentIndex(-1)

    def __choose_patient(self, idx: int, reset_patient: bool = True):
        # triggered by:
        # (1) patient combox update
        # (2) idl_gtvt/gtvncombox update, but can not find cur patient in new idl folder
        if reset_patient is True:
            self.__patient = self.__combox["patient"].currentText()
            self.__reset_zoomin()

        # triggered by idl_gtvt combox update, and find cur patient in new idl folder
        else:
            self.__combox["patient"].setCurrentText(self.__patient)

        # run this after patient combox current text is set up
        self.__enable_arrow_btns("patient")

        # load imgs
        self.__imgs["ct"] = self.__load_img(
            os.path.join(self.__dataset_dir, "HNCDL_{}_CT.nii".format(self.__patient))
        )
        self.__imgs["pt"] = self.__load_img(
            os.path.join(self.__dataset_dir, "HNCDL_{}_PT.nii".format(self.__patient))
        )
        self.__imgs["mrt1"] = self.__load_img(
            os.path.join(self.__dataset_dir, "HNCDL_{}_T1dr.nii".format(self.__patient))
        )
        self.__imgs["mrt2"] = self.__load_img(
            os.path.join(self.__dataset_dir, "HNCDL_{}_T2dr.nii".format(self.__patient))
        )
        # load labels
        if self.__replay_mode:
            for i in ["t", "n"]:
                self.__imgs["gtv{}.label".format(i)] = self.__load_img(
                    os.path.join(
                        self.__dataset_dir,
                        "HNCDL_{}_GTV{}.nii".format(self.__patient, i),
                    )
                )

        # initialize round combobox
        self.__combox["round"].clear()
        self.__arrow_btn["prev.round"].setEnabled(False)
        self.__arrow_btn["next.round"].setEnabled(False)

        round_list = []

        for gtv in ["idl.gtvt", "idl.gtvn"]:
            if self.__combox[gtv].currentText() == "baseline":
                if round_list == []:
                    round_list = ["00"]
                else:
                    pass
            else:
                round_list = ["00", "01"]

        self.__combox["round"].addItems(round_list)

        if self.__replay_mode:
            self.__combox["round"].setEnabled(True)
        else:
            self.__combox["round"].setEnabled(False)

        # get slice id
        slices_count = self.__get_slices_count()
        if slices_count > 0:
            # try to keep slice id unchanged
            # if slice is None, show the middle slice of whole 3D img,
            if self.__slice is None:
                self.__slice = round(slices_count / 2) - 1
            # check slice_id range from [0, slices_count-1]
            self.__slice = ValueUtils.limit_range(self.__slice, (0, slices_count - 1))

        # replay mode
        if self.__replay_mode:
            # try not to reset round when patient is changed
            if self.__round not in round_list:
                # this happens when:
                # (1) current idl training has less round than prev idl training
                # (2) the first time loading a idl training
                reset_round = True
            else:
                reset_round = False
            self.__choose_round(idx=None, reset_round=reset_round)

        # real time mode, always select the newest round
        else:
            self.__round = round_list[-1]
            self.__combox["round"].setCurrentText(self.__round)
            self.__choose_round(idx=None, reset_round=True)
            # update annotation message
            self.__text_box["annotation.msg"].setText(
                "Click the center of the GTVt, then press the 'Confirm' button (green button)"
            )
            # update annotation step
            self.__annotation_step = Step.CLICK_GTVT_CENTER

    def __choose_round(self, idx: int, reset_round: bool = True):
        if reset_round is True:
            self.__round = self.__combox["round"].currentText()
        else:
            self.__combox["round"].setCurrentText(self.__round)

        # run this after "round" combox current text is decided
        if self.__replay_mode:
            self.__enable_arrow_btns("round")
        else:
            self.__arrow_btn["prev.round"].setEnabled(False)
            self.__arrow_btn["next.round"].setEnabled(False)

        # load preds
        path = Dict()
        baseline_id = self.__combox["baseline"].currentText()

        # load gtvt/gtvn preds and gtvn.clicks
        for gtv in ["gtvt", "gtvn"]:
            if (
                self.__round == "00"
                or self.__combox["idl.{}".format(gtv)].currentText() == "baseline"
            ):
                path["{}.pred".format(gtv)] = os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    "baseline",
                    "cross_valid",
                    "patients",
                    "patient={}".format(self.__patient),
                )
                if gtv == "gtvn":
                    path["gtvn.clicks"] = None
            else:
                if gtv == "gtvt":
                    path["{}.pred".format(gtv)] = os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        self.__combox["idl.{}".format(gtv)].currentText(),
                        "patients",
                        "patient={}".format(self.__patient),
                        "round={}".format(self.__round),
                    )
                else:
                    path["gtvn.pred"] = path["gtvn.clicks"] = os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        self.__combox["idl.{}".format(gtv)].currentText(),
                        "cross_valid",
                        "patients",
                        "patient={}".format(self.__patient),
                    )
        # Add the file name to complete the path
        path["gtvt.pred"] = os.path.join(path["gtvt.pred"], "gtvt_pred.nii")
        path["gtvn.pred"] = os.path.join(path["gtvn.pred"], "gtvn_pred.nii")
        if path["gtvn.clicks"] is not None:
            path["gtvn.clicks"] = os.path.join(path["gtvn.clicks"], "gtvn_clicks.nii")

        # load imgs based on paths
        for i in ["gtvt.pred", "gtvn.pred", "gtvn.clicks"]:
            if path[i] is not None and os.path.exists(path[i]):
                self.__imgs[i] = self.__load_img(path[i])
                self.__imgs[i] = Img.binarize(self.__imgs[i])
            else:
                self.__imgs[i] = None

        # load scores on replay mode
        if self.__replay_mode:
            # load baseline gtvt scores
            if (
                self.__combox["idl.gtvt"].currentText() == "baseline"
                or self.__round == "00"
            ):
                gtvt_score = Json.load(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        "baseline",
                        "cross_valid",
                        "inference_test_inter.json",
                    )
                )
                for metric in g.METRICS:
                    self.__scores["gtvt"][metric] = gtvt_score[
                        "patient={}".format(self.__patient)
                    ]["gtvt"][metric]

            # load idl gtvt scores
            else:
                gtvt_score = Json.load(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        self.__combox["idl.gtvt"].currentText(),
                        "inference_test_inter.json",
                    )
                )
                for metric in g.METRICS:
                    self.__scores["gtvt"][metric] = gtvt_score[
                        "patient={}".format(self.__patient)
                    ][metric]["round={}".format(self.__round)]

            # load baseline gtvn scores
            if (
                self.__combox["idl.gtvn"].currentText() == "baseline"
                or self.__round == "00"
            ):
                gtvn_score = Json.load(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        "baseline",
                        "cross_valid",
                        "inference_test_inter.json",
                    )
                )
                for metric in g.METRICS:
                    self.__scores["gtvn"][metric] = gtvn_score[
                        "patient={}".format(self.__patient)
                    ]["gtvn"][metric]

            # load idl gtvn scores
            else:
                gtvn_score = Json.load(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        self.__combox["idl.gtvn"].currentText(),
                        "cross_valid",
                        "inference_test_inter.json",
                    )
                )
                for metric in g.METRICS:
                    self.__scores["gtvn"][metric] = gtvn_score[
                        "patient={}".format(self.__patient)
                    ][metric]["round={}".format(self.__round)]

        else:  # real-time mode, dont show scores
            for i in ["gtvt", "gtvn"]:
                self.__scores[i]["dsc"] = None
                self.__scores[i]["msd"] = None
                self.__scores[i]["hd95"] = None

        # update ui
        self.__refresh_imgs()
        self.__refresh_title()

    def __choose_prev_baseline(self):
        idx = self.__combox["baseline"].currentIndex() - 1
        if idx < 0:
            return
        prev_baseline = self.__combox["baseline"].itemText(idx)
        self.__combox["baseline"].setCurrentText(prev_baseline)
        self.__choose_baseline()

    def __choose_next_baseline(self):
        idx = self.__combox["baseline"].currentIndex() + 1
        if idx > self.__combox["baseline"].count() - 1:
            return
        next_baseline = self.__combox["baseline"].itemText(idx)
        self.__combox["baseline"].setCurrentText(next_baseline)
        self.__choose_baseline()

    def __choose_prev_idl_gtvs(self):
        idx = self.__combox["idl.gtvs"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvs = self.__combox["idl.gtvs"].itemText(idx)
        self.__combox["idl.gtvs"].setCurrentText(prev_idl_gtvs)
        self.__choose_idl_gtvs()

    def __choose_next_idl_gtvs(self):
        idx = self.__combox["idl.gtvs"].currentIndex() + 1
        if idx > self.__combox["idl.gtvs"].count() - 1:
            return
        next_idl_gtvs = self.__combox["idl.gtvs"].itemText(idx)
        self.__combox["idl.gtvs"].setCurrentText(next_idl_gtvs)
        self.__choose_idl_gtvs()

    def __choose_prev_idl_gtvn(self):
        idx = self.__combox["idl.gtvn"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvn = self.__combox["idl.gtvn"].itemText(idx)
        self.__combox["idl.gtvn"].setCurrentText(prev_idl_gtvn)
        self.__choose_idl_gtvn()

    def __choose_next_idl_gtvn(self):
        idx = self.__combox["idl.gtvn"].currentIndex() + 1
        if idx > self.__combox["idl.gtvn"].count() - 1:
            return
        next_idl_gtvn = self.__combox["idl.gtvn"].itemText(idx)
        self.__combox["idl.gtvn"].setCurrentText(next_idl_gtvn)
        self.__choose_idl_gtvn()

    def __choose_prev_idl_gtvt(self):
        idx = self.__combox["idl.gtvt"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvt = self.__combox["idl.gtvt"].itemText(idx)
        self.__combox["idl.gtvt"].setCurrentText(prev_idl_gtvt)
        self.__choose_idl_gtvt()

    def __choose_next_idl_gtvt(self):
        idx = self.__combox["idl.gtvt"].currentIndex() + 1
        if idx > self.__combox["idl.gtvt"].count() - 1:
            return
        next_idl_gtvt = self.__combox["idl.gtvt"].itemText(idx)
        self.__combox["idl.gtvt"].setCurrentText(next_idl_gtvt)
        self.__choose_idl_gtvt()

    def __choose_prev_patient(self):
        idx = self.__combox["patient"].currentIndex() - 1
        if idx < 0:
            return
        prev_patient = self.__combox["patient"].itemText(idx)
        self.__combox["patient"].setCurrentText(prev_patient)
        self.__choose_patient(idx=None, reset_patient=True)

    def __choose_next_patient(self):
        idx = self.__combox["patient"].currentIndex() + 1
        if idx > self.__combox["patient"].count() - 1:
            return
        next_patient = self.__combox["patient"].itemText(idx)
        self.__combox["patient"].setCurrentText(next_patient)
        self.__choose_patient(idx=None, reset_patient=True)

    def __choose_prev_round(self):
        idx = self.__combox["round"].currentIndex() - 1
        if idx < 0:
            return
        prev_round = self.__combox["round"].itemText(idx)
        self.__combox["round"].setCurrentText(prev_round)
        self.__choose_round(idx=None, reset_round=True)

    def __choose_next_round(self):
        idx = self.__combox["round"].currentIndex() + 1
        if idx > self.__combox["round"].count() - 1:
            return
        next_round = self.__combox["round"].itemText(idx)
        self.__combox["round"].setCurrentText(next_round)
        self.__choose_round(idx=None, reset_round=True)

    def __refresh_imgs(self):
        # no img data loaded
        if self.__imgs["pt"] is None:
            return

        # check if cur slice is annotated
        is_annotated = self.__is_annotated()

        # set contour color
        color = Dict()
        color["gtvt.label"] = self.__color["gtvt.label"]
        color["gtvn.label"] = self.__color["gtvn.label"]

        if is_annotated:
            color["gtvt.pred"] = self.__color["gtvt.annotation"]
        else:
            color["gtvt.pred"] = self.__color["gtvt.pred"]

        color["gtvn.pred"] = self.__color["gtvn.pred"]
        color["gtvn.clicks"] = self.__color["gtvn.annotation"]

        # load rgb imgs
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            if self.__img_plane == "sagittal":
                rgb_img = self.__imgs[i][:, :, self.__slice]
            elif self.__img_plane == "coronal":
                rgb_img = self.__imgs[i][:, self.__slice, :]
            # img_plane == "transverse":
            else:
                # for transverse plane, img is upside down,
                # true slice id is: slices_count - 1 - slice_id
                rgb_img = self.__imgs[i][
                    (self.__get_slices_count() - 1 - self.__slice), :, :
                ]

            rgb_img = np.uint8((rgb_img - rgb_img.min()) / rgb_img.ptp() * 255.0)
            # after cv2.cvtColor, rgb_img has 3 channels, but is still numpy
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)

            # update brightness and contrast value when slider bar updated
            if self.__bright_contrast_modality == i:
                self.__bright[i] = self.__slider["bright"].value()
                self.__contrast[i] = self.__slider["contrast"].value()

            # cv2.addWeighted: dst = src1 * alpha + src2 * beta + gamma
            rgb_img = cv2.addWeighted(
                src1=rgb_img,
                alpha=self.__contrast[i] / 100,
                src2=np.zeros_like(rgb_img),
                beta=0,
                gamma=self.__bright[i],
            )

            # add mask to annotated slices
            selected_slices = Dict()
            total_slices_num = Dict()
            if self.__img_plane == "transverse":
                selected_slices["horizontal"] = self.__get_selected_slices("coronal")
                total_slices_num["horizontal"] = self.__imgs["pt"].shape[1]
                selected_slices["vertical"] = self.__get_selected_slices("sagittal")
                total_slices_num["vertical"] = self.__imgs["pt"].shape[2]
            elif self.__img_plane == "coronal":
                selected_slices["horizontal"] = self.__get_selected_slices("transverse")
                total_slices_num["horizontal"] = self.__imgs["pt"].shape[0]
                selected_slices["vertical"] = self.__get_selected_slices("sagittal")
                total_slices_num["vertical"] = self.__imgs["pt"].shape[2]
            elif self.__img_plane == "sagittal":
                selected_slices["horizontal"] = self.__get_selected_slices("transverse")
                total_slices_num["horizontal"] = self.__imgs["pt"].shape[0]
                selected_slices["vertical"] = self.__get_selected_slices("coronal")
                total_slices_num["vertical"] = self.__imgs["pt"].shape[1]

            rgb_img_zeros = np.zeros((rgb_img.shape), dtype=np.uint8)

            selected_slices_mask = None

            # annotated slices mask
            for direction in ["horizontal", "vertical"]:
                for selected_slice in selected_slices[direction]:
                    # image is reversed in the transverse plane
                    if self.__img_plane != "transverse" and direction == "horizontal":
                        slice_pos = total_slices_num[direction] - selected_slice
                    else:
                        slice_pos = selected_slice

                    if direction == "horizontal":
                        x1 = 0
                        y1 = slice_pos
                        x2 = rgb_img.shape[1] - 1
                        y2 = slice_pos
                    elif direction == "vertical":
                        x1 = slice_pos
                        y1 = 0
                        x2 = slice_pos
                        y2 = rgb_img.shape[0] - 1

                    cur_slice_mask = cv2.rectangle(
                        img=rgb_img_zeros,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=self.__color["gtvt.annotation"],
                        thickness=-1,
                    )
                    if selected_slices_mask is None:
                        selected_slices_mask = cur_slice_mask
                    else:
                        selected_slices_mask += cur_slice_mask

            if selected_slices_mask is not None:
                rgb_img = cv2.addWeighted(
                    src1=rgb_img,
                    alpha=1,
                    src2=selected_slices_mask,
                    beta=1,  # 0.5,
                    gamma=0,
                )

            # resize and fit img frame
            rgb_img, self.__resize_pos[i] = self.__fit_display_frame(
                rgb_img, self.__display_frame[i]
            )
            # blur after __fit_display_frame will gain better effect
            rgb_img = cv2.GaussianBlur(rgb_img, (3, 3), cv2.BORDER_DEFAULT)

            # draw label and pred contour
            for k in [
                "gtvt.label",
                "gtvt.pred",
                "gtvn.label",
                "gtvn.pred",
                "gtvn.clicks",
            ]:
                if self.__imgs[k] is None:
                    continue
                else:
                    pass

                if not self.__replay_mode and self.__round == "00":
                    continue
                else:
                    pass

                # load data of current slice
                if self.__img_plane == "sagittal":
                    contours = self.__imgs[k][:, :, self.__slice].astype(np.uint8)
                elif self.__img_plane == "coronal":
                    contours = self.__imgs[k][:, self.__slice, :].astype(np.uint8)
                else:  # img_plane == "transverse":
                    # for transverse plane, img is upside down,
                    # true slice id is: slices_count - 1 - slice_id
                    contours = self.__imgs[k][
                        (self.__get_slices_count() - 1 - self.__slice), :, :
                    ].astype(np.uint8)

                contours, _ = self.__fit_display_frame(
                    contours, self.__display_frame[i]
                )
                # blur after __fit_display_frame will make the contours looks better on the UI
                if k != "gtvn.clicks":
                    contours = cv2.GaussianBlur(contours, (7, 7), cv2.BORDER_DEFAULT)
                contours, _ = cv2.findContours(
                    contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                if k == "gtvn.clicks":
                    thickness = 7
                else:
                    thickness = 2
                rgb_img = cv2.drawContours(
                    image=rgb_img,
                    contours=contours,
                    contourIdx=-1,
                    color=color[k],
                    thickness=thickness,
                )

            rgb_img_height = rgb_img.shape[0]
            rgb_img_width = rgb_img.shape[1]
            rgb_img_chan = rgb_img.shape[2]

            # add score text
            if self.__replay_mode:
                cv_text = ""
                text_pos_x = 10
                text_pos_y = 10

                for metric in g.METRICS:
                    cv_text += "GTVt - " + metric.upper()
                    if ValueUtils.is_number(self.__scores["gtvt"][metric]):
                        cv_text += ": {:.3f}".format(self.__scores["gtvt"][metric])
                    else:
                        cv_text += ": N/A"
                    cv_text += "\n"
                for metric in g.METRICS:
                    cv_text += "GTVn - " + metric.upper()
                    if ValueUtils.is_number(self.__scores["gtvn"][metric]):
                        cv_text += ": {:.3f}".format(self.__scores["gtvn"][metric])
                    else:
                        cv_text += ": N/A"
                    cv_text += "\n"

                self.__cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(text_pos_x, text_pos_y),
                    color=self.__color["score.text"],
                )

                # add text label gtvt
                text_pos_y = rgb_img_height - 108
                # text_pos_y = height - 46
                text_pos_gap = 20

                cv_text = "LABEL - GTVt"
                self.__cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(text_pos_x, text_pos_y),
                    color=color["gtvt.label"],
                )

                # add text pred gtvt
                text_pos_y += text_pos_gap
                cv_text = "PRED - GTVt"
                if is_annotated:
                    cv_text += " (ANNOTATED)"
                self.__cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(text_pos_x, text_pos_y),
                    color=color["gtvt.pred"],
                )

                # add text label gtvn
                text_pos_y += text_pos_gap
                cv_text = "LABEL - GTVn"
                self.__cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(text_pos_x, text_pos_y),
                    color=color["gtvn.label"],
                )

                # add text pred gtvn
                text_pos_y += text_pos_gap
                cv_text = "PRED - GTVn"
                self.__cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(text_pos_x, text_pos_y),
                    color=color["gtvn.pred"],
                )

                # add text label gtvn
                text_pos_y += text_pos_gap
                cv_text = "CLICKS - GTVn"
                self.__cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(text_pos_x, text_pos_y),
                    color=color["gtvn.clicks"],
                )

            # show imgs
            qt_image = QImage(
                rgb_img,
                rgb_img_width,
                rgb_img_height,
                rgb_img_width * rgb_img_chan,
                QImage.Format_RGB888,
            )
            self.__display_frame[i].setPixmap(QPixmap.fromImage(qt_image))

    def __get_selected_slices(self, plane) -> List:
        # get current round

        if self.__round == "00":
            return []

        # load annotated slices
        baseline_id = self.__combox["baseline"].currentText()
        idl_gtvt_id = self.__combox["idl.gtvt"].currentText()
        idl_gtvt_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvt_id)
        json_path = os.path.join(
            idl_gtvt_dir,
            "patients",
            "patient={}".format(self.__patient),
            "selected_slices.json",
        )
        if not os.path.exists(json_path):
            return []

        selected_slices_dict = Json.load(json_path)[plane]
        selected_slices_list = List()

        for round_num in selected_slices_dict:
            selected_slices_list += List(selected_slices_dict[round_num])

            if (round_num[len("round=") :]) == self.__round:
                break

        # change annotated slice from str to int
        for i in range(len(selected_slices_list)):
            selected_slices_list[i] = int(selected_slices_list[i])

        return selected_slices_list

    def __is_annotated(self) -> bool:
        if self.__imgs["pt"] is None:
            return False

        if int(self.__slice) in self.__get_selected_slices(self.__img_plane):
            return True
        else:
            return False

    def __cv_put_text(
        self,
        img,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        line_gap: int = 20,
    ):
        for i, line in enumerate(text.split("\n")):
            y = pos[1] + i * line_gap
            y += 15
            cv2.putText(
                img=img,
                text=line,
                org=(pos[0], y),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.0,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def wheelEvent(self, event):
        super().wheelEvent(event)
        if self.__slice is not None:
            slices_count = self.__get_slices_count()
            if slices_count == 0:
                return
            slice_delta = event.angleDelta().y() // 120
            if self.__img_plane == "coronal":
                slice_delta = -slice_delta
            self.__slice -= slice_delta
            # limite slice_id in range(0,slices_count)
            self.__slice %= slices_count
            self.__refresh_imgs()
            self.__refresh_title()

    def __get_slices_count(self) -> int:
        if (self.__imgs["pt"] is None) or (self.__img_plane is None):
            return 0
        if self.__img_plane == "sagittal":
            slices_count = self.__imgs["pt"].shape[2]
        elif self.__img_plane == "coronal":
            slices_count = self.__imgs["pt"].shape[1]
        else:  # img_plane == "transverse"
            slices_count = self.__imgs["pt"].shape[0]
        return slices_count

    def __refresh_title(self):
        win_tital = "iDL.Tool "
        if self.__round is not None:
            win_tital += "   Num.of.Annotated.Slices="
            win_tital += str(len(self.__get_selected_slices(self.__img_plane)))
        if self.__slice is not None:
            slices_count = self.__get_slices_count()
            if slices_count > 0:
                win_tital += "   Slice={}/{}".format(self.__slice + 1, slices_count)
        self.setWindowTitle(win_tital)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__resize_display_frames()
        self.__refresh_side_bar()
        self.__refresh_imgs()

    def __resize_display_frames(self):
        gap = 1
        size = Dict()
        size["x"] = self.geometry().width() - self.__side_bar_width
        size["y"] = self.geometry().height()
        for i in ["x", "y"]:
            double_size = size[i] - gap * 3
            size.pop(i)
            size[i][0] = double_size // 2
            size[i][1] = double_size // 2
            if double_size % 2 != 0:
                size[i][0] += 1

        pos = Dict()
        for i in ["x", "y"]:
            pos[i][0] = gap
            pos[i][1] = size[i][0] + gap * 2

        self.__display_frame["ct"].setGeometry(
            QRect(pos["x"][0], pos["y"][0], size["x"][0], size["y"][0])
        )
        self.__display_frame["pt"].setGeometry(
            QRect(pos["x"][1], pos["y"][0], size["x"][1], size["y"][0])
        )
        self.__display_frame["mrt1"].setGeometry(
            QRect(pos["x"][0], pos["y"][1], size["x"][0], size["y"][1])
        )
        self.__display_frame["mrt2"].setGeometry(
            QRect(pos["x"][1], pos["y"][1], size["x"][1], size["y"][1])
        )

    def __open_file_dlg(self):
        Tk().withdraw()
        file_name = filedialog.askopenfilename()
        if file_name == "" or file_name is None:
            pass


app = QApplication(sys.argv)
ui_core = UiCore()
ui_core.show()
sys.exit(app.exec_())
