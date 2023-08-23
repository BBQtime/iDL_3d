import os
import platform
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Tuple

import cv2
import numpy as np
from custom import Debug, Dict, Directory
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtWidgets import QApplication, QButtonGroup, QMainWindow
from Ui_core import Ui_Core

# gravity center of gtvs


class UiReplay(QMainWindow, Ui_Core):
    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.setupUi(self)

        self._init_ui_names()
        self._init_member_var(debug_mode)
        self.__set_display_frames_background()
        # self.__init_zoomin()
        self.__init_color()
        self._init_side_bar()  # after _init_member_var(), function connection needed

        self._clear_img_data()
        self._refresh_title()  # after _init_member_var()

        # resize
        self.resize(1200, 800)  # set origin size
        self.showMaximized()

        # load first baseline result
        self._choose_baseline()

    def _init_member_var(
        self,
        debug_mode: bool = False,  # param: debug_mode is for subclass: UiIdl
    ):
        # load test set patients of au and mda datasets
        # DATASET_SPLIT_JSON_PATH["au.1mm"] and ["au.3mm"] are the same
        dataset_split_au = Json.load(g.DATASET_SPLIT_JSON_PATH["au.1mm"])
        dataset_split_mda = Json.load(g.DATASET_SPLIT_JSON_PATH["mda"])
        self.__patients = Dict()
        self.__patients["au.test.inter"] = List(dataset_split_au["test.inter"])
        self.__patients["au.test.exter"] = List(dataset_split_au["test.exter"])
        self.__patients["mda.test"] = List(dataset_split_mda["test"])

        self._baseline_id = None
        self._cur_patient = None
        self._cur_slice = 0  # starts from 0

        self._idl_id = Dict()
        self._idl_round = Dict()
        for i in ["gtvt", "gtvn"]:
            self._idl_id[i] = "baseline"
            self._idl_round[i] = "round=00"

        self.__dataset_ver = None
        self.__dataset_section = None
        self._nii_spacing = None  # (1,1,1) or (1,1,3)
        self._dataset_dir = None  # au.1mm / au.1mm / mda
        self.__scores = Dict()
        self._3d_imgs = Dict()
        self.__resize_pos = Dict()
        self.__side_bar_width = 300

    def _clear_img_data(self):
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
            self._3d_imgs[i] = None

        # resize position of ct/pt/mrt1/mrt2
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__resize_pos[i] = None

        # set image plane
        for i in ["transverse", "coronal", "sagittal"]:
            if self.__radio_btn[i].isChecked():
                self.__img_plane = i

    # def __init_zoomin(self):
    #     self.__zoomin = Dict()
    #     self.__zoomin["rubber.band"] = QRubberBand(QRubberBand.Rectangle, self)
    #     self._reset_zoomin()

    # def _reset_zoomin(self):
    #     self.__zoomin["rubber.band"].hide()
    #     self.__zoomin["img"] = None
    #     self.__zoomin["start"] = None
    #     self.__zoomin["end"] = None

    # def mousePressEvent(self, event):
    #     super().mousePressEvent(event)

    #     # loop 4 img frames
    #     for i in ["ct", "pt", "mrt1", "mrt2"]:
    #         left = self._display_frame[i].x()
    #         top = self._display_frame[i].y()
    #         width = self._display_frame[i].width()
    #         height = self._display_frame[i].height()
    #         # if start pos is in current img frame
    #         if (
    #             event.x() >= left
    #             and event.x() <= left + width
    #             and event.y() >= top
    #             and event.y() <= top + height
    #         ):
    #             # already zoomed in, clear zoomin (only click in img frame area)
    #             if self.__zoomin["start"] is not None:
    #                 self._reset_zoomin()
    #                 self._refresh_imgs()
    #                 return
    #             # zoom in
    #             else:
    #                 self.__zoomin["img"] = i
    #                 self.__zoomin["start"] = event.pos()
    #                 rect = QRect(event.pos(), event.pos())
    #                 self.__zoomin["rubber.band"].setGeometry(rect.normalized())
    #                 self.__zoomin["rubber.band"].show()
    #                 return

    # def mouseMoveEvent(self, event):
    #     super().mouseMoveEvent(event)
    #     if self.__zoomin["start"] is None:  # or self.__zoomin["rubber.band"] is None:
    #         return
    #     self.__mouse_move_event(event)

    # def __mouse_move_event(self, event):
    #     # limit zoomin frame in img frame
    #     display_frame = self._display_frame[self.__zoomin["img"]]
    #     display_frame_right = display_frame.x() + display_frame.width() - 1
    #     if event.x() < display_frame.x():
    #         event_x = display_frame.x()
    #     elif event.x() > display_frame_right:
    #         event_x = display_frame_right
    #     else:
    #         event_x = event.x()
    #     display_frame_buttom = display_frame.y() + display_frame.height() - 1
    #     if event.y() < display_frame.y():
    #         event_y = display_frame.y()
    #     elif event.y() > display_frame_buttom:
    #         event_y = display_frame_buttom
    #     else:
    #         event_y = event.y()
    #     # resize zoomin frame
    #     self.__zoomin["end"] = QPoint(event_x, event_y)
    #     rect = QRect(
    #         self.__zoomin["start"],
    #         self.__zoomin["end"],
    #     )
    #     self.__zoomin["rubber.band"].setGeometry(rect.normalized())

    # def mouseReleaseEvent(self, event):
    #     super().mouseReleaseEvent(event)

    #     # not zoomed in
    #     if self.__zoomin["start"] is None:  # or self.__zoomin["rubber.band"] is None:
    #         return
    #     self.__mouse_move_event(event)
    #     self.__zoomin["rubber.band"].hide()
    #     # self.__zoomin["rubber.band"] = None

    #     # no data loaded
    #     if self._3d_imgs["pt"] is None:
    #         self._reset_zoomin()
    #         return

    #     # zoomin size == 0
    #     if (
    #         abs(self.__zoomin["start"].x() - self.__zoomin["end"].x()) <= 1
    #         or abs(self.__zoomin["start"].y() - self.__zoomin["end"].y()) <= 1
    #     ):
    #         # print("zoomin size 0")
    #         self._reset_zoomin()
    #         return
    #     self.__get_img_roi()

    # def __get_img_roi(self):
    #     # make sure start point always < end point
    #     start_x = self.__zoomin["start"].x()
    #     start_y = self.__zoomin["start"].y()
    #     end_x = self.__zoomin["end"].x()
    #     end_y = self.__zoomin["end"].y()
    #     if start_x > end_x:
    #         x = start_x
    #         start_x = end_x
    #         end_x = x
    #     if start_y > end_y:
    #         y = start_y
    #         start_y = end_y
    #         end_y = y
    #     # get display_frame related position
    #     display_frame_left = self._display_frame[self.__zoomin["img"]].x()
    #     display_frame_top = self._display_frame[self.__zoomin["img"]].y()
    #     start_x -= display_frame_left
    #     end_x -= display_frame_left
    #     start_y -= display_frame_top
    #     end_y -= display_frame_top

    #     # get actual_img_area related position
    #     resize_pos = self.__resize_pos[self.__zoomin["img"]]
    #     start_x -= resize_pos["x"]
    #     start_y -= resize_pos["y"]
    #     end_x -= resize_pos["x"]
    #     end_y -= resize_pos["y"]
    #     # out of range
    #     if (start_x < 0 and end_x < 0) or (
    #         start_x > resize_pos["width"] and end_x > resize_pos["width"]
    #     ):
    #         self._reset_zoomin()
    #         return
    #     if (start_y < 0 and end_y < 0) or (
    #         start_y > resize_pos["height"] and end_y > resize_pos["height"]
    #     ):
    #         self._reset_zoomin()
    #         return
    #     # limit zoomin frame in image area
    #     if start_x < 0:
    #         start_x = 0
    #     if start_y < 0:
    #         start_y = 0
    #     if end_x > resize_pos["width"]:
    #         end_x = resize_pos["width"]
    #     if end_y > resize_pos["height"]:
    #         end_y = resize_pos["height"]

    #     # get actual zoom position
    #     if self.__img_plane == "sagittal":
    #         origin_width = self._3d_imgs["pt"].shape[1]
    #         origin_height = self._3d_imgs["pt"].shape[0]
    #         origin_height = round(
    #             origin_height * self._nii_spacing[2] / self._nii_spacing[1]
    #         )
    #     elif self.__img_plane == "coronal":
    #         origin_width = self._3d_imgs["pt"].shape[2]
    #         origin_height = self._3d_imgs["pt"].shape[0]
    #         origin_height = round(
    #             origin_height * self._nii_spacing[2] / self._nii_spacing[0]
    #         )
    #     else:
    #         origin_width = self._3d_imgs["pt"].shape[2]
    #         origin_height = self._3d_imgs["pt"].shape[1]

    #     start_x = round(start_x * origin_width / resize_pos["width"])
    #     end_x = round(end_x * origin_width / resize_pos["width"])
    #     start_y = round(start_y * origin_height / resize_pos["height"])
    #     end_y = round(end_y * origin_height / resize_pos["height"])

    #     self.__zoomin["start"] = QPoint(start_x, start_y)
    #     self.__zoomin["end"] = QPoint(end_x, end_y)
    #     self._refresh_imgs()

    def __fit_display_frame(self, img, display_frame: QtWidgets.QLabel):
        err_msg = "MainWindow.__fit_display_frame(), img.shape should == 2 or 3"

        # image spacing resize
        if self.__img_plane == "sagittal":
            spacing_height = round(
                img.shape[0] * self._nii_spacing[2] / self._nii_spacing[1]
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
                img.shape[0] * self._nii_spacing[2] / self._nii_spacing[0]
            )
            img = cv2.resize(
                img,
                (
                    img.shape[1],
                    spacing_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        # # zoom in
        # if self.__zoomin["start"] is not None and self.__zoomin["end"] is not None:
        #     if len(img.shape) == 3:
        #         img = img[
        #             self.__zoomin["start"].y() : self.__zoomin["end"].y(),
        #             self.__zoomin["start"].x() : self.__zoomin["end"].x(),
        #             :,
        #         ]
        #     elif len(img.shape) == 2:
        #         img = img[
        #             self.__zoomin["start"].y() : self.__zoomin["end"].y(),
        #             self.__zoomin["start"].x() : self.__zoomin["end"].x(),
        #         ]
        #     else:
        #         raise ValueError(err_msg)

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
        self._color = Dict()
        self._color["gtvt.label"] = (0, 255, 255)  # light blue
        self._color["gtvn.label"] = (0, 150, 255)  # dark blue
        self._color["gtvt.pred"] = (255, 255, 0)  # yellow
        self._color["gtvn.pred"] = (255, 128, 0)  # orange
        self._color["gtvt.annotation"] = (0, 255, 64)  # green
        self._color["gtvn.annotation"] = (255, 70, 200)  # pink
        self._color["score.text"] = self._color["gtvt.annotation"]

    def _init_ui_names(self):
        self._display_frame = Dict()
        self._display_frame["ct"] = self._display_frame_ct
        self._display_frame["pt"] = self._display_frame_pt
        self._display_frame["mrt1"] = self._display_frame_mrt1
        self._display_frame["mrt2"] = self._display_frame_mrt2

        self._text_label = Dict()
        self._text_label["baseline"] = self._text_label_baseline
        self._text_label["patient"] = self._text_label_patient
        self._text_label["idl.gtvt"] = self._text_label_idl_gtvt
        self._text_label["idl.gtvn"] = self._text_label_idl_gtvn
        self._text_label["bright"] = self._text_label_bright
        self._text_label["contrast"] = self._text_label_contrast
        self._text_label["zoom"] = self._text_label_zoom
        self._text_label["annotation.tools"] = self._text_label_annotation_tools
        self._text_label["idl.gtvt.progress"] = self._text_label_idl_gtvt_progress

        self._combox = Dict()
        self._combox["baseline"] = self._combox_baseline
        self._combox["patient"] = self._combox_patient
        self._combox["idl.gtvt"] = self._combox_idl_gtvt
        self._combox["idl.gtvn"] = self._combox_idl_gtvn

        self._arrow_btn = Dict()
        self._arrow_btn["prev.baseline"] = self._btn_prev_baseline
        self._arrow_btn["next.baseline"] = self._btn_next_baseline
        self._arrow_btn["prev.patient"] = self._btn_prev_patient
        self._arrow_btn["next.patient"] = self._btn_next_patient
        self._arrow_btn["prev.idl.gtvt"] = self._btn_prev_idl_gtvt
        self._arrow_btn["next.idl.gtvt"] = self._btn_next_idl_gtvt
        self._arrow_btn["prev.idl.gtvn"] = self._btn_prev_idl_gtvn
        self._arrow_btn["next.idl.gtvn"] = self._btn_next_idl_gtvn

        self.__radio_btn = Dict()
        self.__radio_btn["ct"] = self._radio_btn_ct
        self.__radio_btn["pt"] = self._radio_btn_pt
        self.__radio_btn["mrt1"] = self._radio_btn_mrt1
        self.__radio_btn["mrt2"] = self._radio_btn_mrt2
        self.__radio_btn["transverse"] = self._radio_btn_transverse
        self.__radio_btn["coronal"] = self._radio_btn_coronal
        self.__radio_btn["sagittal"] = self._radio_btn_sagittal

        self.__slider = Dict()
        self.__slider["bright.ct"] = self._slider_bright_ct
        self.__slider["bright.pt"] = self._slider_bright_pt
        self.__slider["bright.mrt1"] = self._slider_bright_mrt1
        self.__slider["bright.mrt2"] = self._slider_bright_mrt2
        self.__slider["contrast.ct"] = self._slider_contrast_ct
        self.__slider["contrast.pt"] = self._slider_contrast_pt
        self.__slider["contrast.mrt1"] = self._slider_contrast_mrt1
        self.__slider["contrast.mrt2"] = self._slider_contrast_mrt2
        self.__slider["zoom"] = self._slider_zoom

    # set display frames background black
    def __set_display_frames_background(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, Qt.black)
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self._display_frame[i].setObjectName("")
            self._display_frame[i].setAutoFillBackground(True)
            self._display_frame[i].setPalette(pal)

    def _init_side_bar(self):
        # hide idl.gtvs controls
        self._text_label_idl_gtvs.hide()
        self._combox_idl_gtvs.hide()
        self._btn_prev_idl_gtvs.hide()
        self._btn_next_idl_gtvs.hide()

        # hide annotation controls
        self._text_box_annotation_msg.hide()
        self._text_label_annotation_tools.hide()
        self._btn_pen.hide()
        self._btn_eraser.hide()
        self._btn_clear.hide()
        self._btn_confirm.hide()
        self._text_label_idl_gtvt_progress.hide()
        self._progress_bar_idl_gtvt.hide()

        # set text
        self._text_label["baseline"].setText("Choose Baseline")
        self._text_label["patient"].setText("Choose Patient")
        self._text_label["idl.gtvt"].setText("Choose iDL GTVt")
        self._text_label["idl.gtvn"].setText("Choose iDL GTVn")

        self._text_label["bright"].setText("Brightness (CT)")
        self._text_label["contrast"].setText("Contrast (CT)")
        self._text_label["zoom"].setText("Zoom In")

        self.__radio_btn["ct"].setText("CT")
        self.__radio_btn["pt"].setText("PT")
        self.__radio_btn["mrt1"].setText("MR-T1")
        self.__radio_btn["mrt2"].setText("MR-T2")
        self.__radio_btn["transverse"].setText("Transverse")
        self.__radio_btn["coronal"].setText("Coronal")
        self.__radio_btn["sagittal"].setText("Sagittal")

        # set font
        self._font_bold = self._text_label["baseline"].font()
        self._font_light = self._text_label["baseline"].font()
        self._font_bold.setPointSize(8)
        self._font_light.setPointSize(8)
        self._font_bold.setBold(True)
        self._font_light.setBold(False)

        # set font of text labels
        for i in [
            "baseline",
            "patient",
            "idl.gtvt",
            "idl.gtvn",
            "bright",
            "contrast",
            "zoom",
        ]:
            self._text_label[i].setFont(self._font_bold)

        # set font of radio buttons
        for i in ["transverse", "coronal", "sagittal", "ct", "pt", "mrt1", "mrt2"]:
            self.__radio_btn[i].setFont(self._font_bold)

        # set font of comboboxes
        for i in ["baseline", "patient", "idl.gtvt", "idl.gtvn"]:
            self._combox[i].setFont(self._font_light)

        # set combobox dropdown width: 700px
        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._combox[i].setStyleSheet(
                """*
                QComboBox QAbstractItemView
                {
                    min-width: 500px;
                }
                """
            )

        # fill the baseline combobox with baseline_ids
        self._fill_combox_baseline()

        # set initial state
        for i in ["baseline", "patient", "idl.gtvt", "idl.gtvn"]:
            self._arrow_btn["prev.{}".format(i)].setArrowType(Qt.LeftArrow)
            self._arrow_btn["next.{}".format(i)].setArrowType(Qt.RightArrow)

        for i in ["patient", "idl.gtvt", "idl.gtvn"]:
            self._combox[i].setEnabled(False)
            self._arrow_btn["prev.{}".format(i)].setEnabled(False)
            self._arrow_btn["next.{}".format(i)].setEnabled(False)

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
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__slider["bright.{}".format(i)].setMinimum(-128)
            self.__slider["bright.{}".format(i)].setMaximum(128)
            self.__slider["bright.{}".format(i)].setValue(0)
            self.__slider["contrast.{}".format(i)].setMinimum(0)
            self.__slider["contrast.{}".format(i)].setMaximum(200)
            self.__slider["contrast.{}".format(i)].setValue(100)

        # only show ct bright/contrast slider bars
        for i in ["pt", "mrt1", "mrt2"]:
            self.__slider["bright.{}".format(i)].hide()
            self.__slider["contrast.{}".format(i)].hide()

        self.__slider["zoom"].setMinimum(100)
        self.__slider["zoom"].setMaximum(200)
        self.__slider["zoom"].setValue(100)

        # connect ui to functions
        # (put the connections at last, because these functions will need the initialization above)
        self._combox["baseline"].activated.connect(self._choose_baseline)
        self._arrow_btn["prev.baseline"].clicked.connect(self.__choose_prev_baseline)
        self._arrow_btn["next.baseline"].clicked.connect(self.__choose_next_baseline)

        self._combox["patient"].activated.connect(self._choose_patient)
        self._arrow_btn["prev.patient"].clicked.connect(self.__choose_prev_patient)
        self._arrow_btn["next.patient"].clicked.connect(self.__choose_next_patient)

        self._combox["idl.gtvt"].activated.connect(self._choose_idl_gtvt)
        self._arrow_btn["prev.idl.gtvt"].clicked.connect(self.__choose_prev_idl_gtvt)
        self._arrow_btn["next.idl.gtvt"].clicked.connect(self.__choose_next_idl_gtvt)

        self._combox["idl.gtvn"].activated.connect(self._choose_idl_gtvn)
        self._arrow_btn["prev.idl.gtvn"].clicked.connect(self.__choose_prev_idl_gtvn)
        self._arrow_btn["next.idl.gtvn"].clicked.connect(self.__choose_next_idl_gtvn)

        for i in ["bright", "contrast"]:
            for j in ["ct", "pt", "mrt1", "mrt2"]:
                self.__slider["{}.{}".format(i, j)].valueChanged.connect(
                    self._refresh_imgs
                )

        self.__btn_group_plane.buttonClicked.connect(self.__set_img_plane)

        self.__btn_group_bright_contrast.buttonClicked.connect(
            self.__set_bright_contrast_modality
        )

    def _fill_combox_baseline(self):
        baseline_id_list = Directory.get_sub_folders(
            g.TRAIN_RESULTS_DIR, key_word="baseline_", shuffle=False
        )
        self._combox["baseline"].addItems(baseline_id_list)

    def _refresh_side_bar(
        self, widgets_to_display: list = ["baseline", "patient", "idl.gtvt", "idl.gtvn"]
    ):
        left = 30
        top = 0
        text_height = 25
        bar_height = 25
        slider_height = 20
        arrow_btn_width = 30

        if platform.system().lower() == "linux":
            gap = 30
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

        # set position of text label / comboxes / btns
        for i in widgets_to_display:
            # text label
            top += gap
            rect = QRect(left, top, width, text_height)
            self._text_label[i].setGeometry(rect)
            top += text_height

            # btn prev
            tmp_left = left
            rect = QRect(tmp_left, top, arrow_btn_width, bar_height)
            self._arrow_btn["prev.{}".format(i)].setGeometry(rect)

            # combobox
            tmp_left += arrow_btn_width
            rect = QRect(tmp_left + 1, top, width - arrow_btn_width * 2 - 2, bar_height)
            self._combox[i].setGeometry(rect)

            # btn next
            tmp_left += width - arrow_btn_width * 2
            rect = QRect(tmp_left, top, arrow_btn_width, bar_height)
            self._arrow_btn["next.{}".format(i)].setGeometry(rect)

            # next element
            top += bar_height

        # brightness and contrast sliders
        top += gap
        for i in ["bright", "contrast"]:
            rect = QRect(left, top, width, text_height)
            self._text_label[i].setGeometry(rect)
            top += text_height
            rect = QRect(left, top, width, slider_height)
            for j in ["ct", "pt", "mrt1", "mrt2"]:
                self.__slider["{}.{}".format(i, j)].setGeometry(rect)
            top += slider_height

        # brightness and contrast radio btns
        tmp_left = left
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            rect = QRect(tmp_left, top, radio_btn_width[i], radio_btn_height)
            self.__radio_btn[i].setGeometry(rect)
            tmp_left += radio_btn_gap["bright.contrast"] + radio_btn_width[i]
        top += radio_btn_height

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
        self._text_label["zoom"].setGeometry(rect)
        top += text_height
        rect = QRect(left, top, width, slider_height)
        self.__slider["zoom"].setGeometry(rect)
        top += slider_height

        # return the followings for UiIdl
        return left, top, width, gap, text_height, bar_height

    def __set_img_plane(self):
        for i in ["transverse", "coronal", "sagittal"]:
            if self.__radio_btn[i].isChecked():
                self.__img_plane = i
                break

        # update and check slice_id (starts from 0)
        slices_count = self.__get_slices_count()
        if slices_count > 0:
            self._cur_slice = round(slices_count / 2) - 1
            self._cur_slice = Value.limit_range(self._cur_slice, (0, slices_count - 1))
        # self._reset_zoomin()
        self._refresh_imgs()
        self._refresh_title()

    def __set_bright_contrast_modality(self):
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            if self.__radio_btn[i].isChecked():
                self.__slider["bright.{}".format(i)].show()
                self.__slider["contrast.{}".format(i)].show()
            else:
                self.__slider["bright.{}".format(i)].hide()
                self.__slider["contrast.{}".format(i)].hide()

        if self.__radio_btn["ct"].isChecked():
            key_word = "CT"
        elif self.__radio_btn["pt"].isChecked():
            key_word = "PT"
        elif self.__radio_btn["mrt1"].isChecked():
            key_word = "MR-T1"
        elif self.__radio_btn["mrt2"].isChecked():
            key_word = "MR-T2"
        else:
            Debug.error_exit("no radio button is checked")

        self._text_label["bright"].setText("Brightness ({})".format(key_word))
        self._text_label["contrast"].setText("Contrast ({})".format(key_word))

    def _clear_display_frames(self):
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            width = self._display_frame[i].width()
            height = self._display_frame[i].height()
            black_img = np.zeros([width, height, 3])
            qt_image = QImage(
                black_img,
                self._display_frame[i].width(),
                self._display_frame[i].height(),
                self._display_frame[i].width() * 3,
                QImage.Format_RGB888,
            )
            self._display_frame[i].setPixmap(QPixmap.fromImage(qt_image))

    def _enable_arrow_btns(self, combox_name: str):
        # enable/disable prev/next round buttons
        idx = self._combox[combox_name].currentIndex()
        if idx == 0:
            self._arrow_btn["prev.{}".format(combox_name)].setEnabled(False)
        else:
            self._arrow_btn["prev.{}".format(combox_name)].setEnabled(True)

        if idx == (self._combox[combox_name].count() - 1):
            self._arrow_btn["next.{}".format(combox_name)].setEnabled(False)
        else:
            self._arrow_btn["next.{}".format(combox_name)].setEnabled(True)

    def _get_combox_contents(self, combox: QtWidgets.QComboBox):
        content_list = List()
        for i in range(combox.count()):
            content_list.append(combox.itemText(i))
        return content_list

    def _load_dataset_dir_and_nii_spacing(self):
        # load slice thickness from baseline hyper
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, "baseline")
        fold_dir = Directory.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        )[0]
        baseline_dataset_ver = Json.load(os.path.join(fold_dir, "hyper.json"))[
            "dataset.ver"
        ]

        # set dataset dir based on current patient
        if self._cur_patient in self.__patients["au.test.inter"]:
            self.__dataset_ver = baseline_dataset_ver
            self.__dataset_section = "test.inter"

        elif self._cur_patient in self.__patients["au.test.exter"]:
            self.__dataset_ver = baseline_dataset_ver
            self.__dataset_section = "test.exter"

        elif self._cur_patient in self.__patients["mda.test"]:
            self.__dataset_ver = "mda"
            self.__dataset_section = "test"
        else:
            Debug.error_exit("cant find current patient in test patients")

        # set dataset dir and nii spacing
        self._dataset_dir = g.DATASET_DIR[self.__dataset_ver]
        self._nii_spacing = g.NII_SPACING[self.__dataset_ver]

    def _fill_combox_patient(self):
        combox_patients = Directory.get_sub_folders(
            os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, "baseline", "patients")
        )
        # from "patient=123" to "123"
        for i in range(len(combox_patients)):
            combox_patients[i] = combox_patients[i][len("patient=") :]

        combox_patients.find_identical_items(self.__patients.to_list())
        combox_patients.sort()
        self._combox["patient"].addItems(combox_patients)
        self._combox["patient"].setEnabled(True)
        return combox_patients

    def _choose_baseline(self):
        # self._reset_zoomin()
        self._clear_img_data()
        self._clear_display_frames()

        # run this after current text of baseline combox is confirmed
        self._enable_arrow_btns("baseline")

        self._baseline_id = self._combox["baseline"].currentText()

        # reset comboboxes
        for i in ["patient", "idl.gtvt", "idl.gtvn"]:
            self._combox[i].clear()
            self._combox[i].setEnabled(False)
            self._arrow_btn["prev.{}".format(i)].setEnabled(False)
            self._arrow_btn["next.{}".format(i)].setEnabled(False)

        # run this after self._baseline_id is confirmed
        combox_patients = self._fill_combox_patient()

        # choose patient automatically
        # try not to reset patient when idl.gtvt/gtvn_id are changed
        if self._cur_patient not in combox_patients:
            reset_patient = True
        else:
            reset_patient = False
        self._choose_patient(idx=None, reset_patient=reset_patient)

    def __load_3d_img(self, path: str, binary: bool = False):
        img = Nii.load(path, binary=False)

        # ct windowing before normalization
        if "CT" in path:
            img = Img.ct_windowing(img)

        # normalization
        img = Img.normalize(img)

        # binarize img after normalization
        if binary:
            img = Img.binarize(img)

        # turn upside down
        img = np.flip(img, axis=0)

        # flip left/right for 1mm data
        if self._nii_spacing[2] == 1.0:
            img = np.flip(img, axis=2)

        return img

    # ui_idl will inherit this function, do not make it a private function
    def _load_multi_modal_imgs(self):
        paths = Dict()
        paths["ct"] = "CT"
        paths["pt"] = "PT"
        paths["mrt1"] = "T1dr"
        paths["mrt2"] = "T2dr"
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            paths[i] = "HNCDL_{}_{}.nii".format(self._cur_patient, paths[i])
            paths[i] = os.path.join(self._dataset_dir, paths[i])
            self._3d_imgs[i] = self.__load_3d_img(paths[i])

    def _get_middle_slice_id(self):
        slices_count = self.__get_slices_count()
        if slices_count > 0:
            # show the middle slice of whole 3D img,
            slice_id = round(slices_count / 2) - 1
            slice_id = Value.limit_range(slice_id, (0, slices_count - 1))
            return slice_id
        else:
            return None

    def _choose_patient(self, idx: int = None, reset_patient: bool = True):
        # triggered by:
        # (1) patient combox update
        # (2) baseline combox update, but can not find cur patient in new baseline dir
        if reset_patient is True:
            self._cur_patient = self._combox["patient"].currentText()
            # self._reset_zoomin()

        # triggered by baseline combox update, and find cur patient in new baseline dir
        else:
            self._combox["patient"].setCurrentText(self._cur_patient)

        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_dir_and_nii_spacing()

        # reset comboboxes
        for i in ["idl.gtvt", "idl.gtvn"]:
            self._combox[i].clear()
            self._combox[i].setEnabled(False)
            self._arrow_btn["prev.{}".format(i)].setEnabled(False)
            self._arrow_btn["next.{}".format(i)].setEnabled(False)

        # fill idl comboboxes
        for i in ["idl.gtvt", "idl.gtvn"]:
            combox_items = ["baseline"]
            # get all round folder under current patient folder
            for idl_result_dir in Directory.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id),
                key_word=i,
                full_path=True,
            ):
                patient_dir = os.path.join(
                    idl_result_dir,
                    "patients",
                    "patient={}".format(self._cur_patient),
                )
                if os.path.exists(patient_dir):
                    round_folders = Directory.get_sub_folders(
                        patient_dir,
                        key_word="round=",
                        full_path=False,
                    )
                    for round_folder in round_folders:
                        combox_items.append(
                            os.path.join(Path(idl_result_dir).name, round_folder)
                        )

            self._combox[i].addItems(combox_items)

            # enable idl.gtvt/gtvn combobox
            self._combox[i].setEnabled(True)
            self._enable_arrow_btns(i)

            # no idl found, show baseline
            if self._combox[i].count() == 1:
                self._combox[i].setCurrentIndex(0)
            # otherwise, show first idl result
            else:
                self._combox[i].setCurrentIndex(1)

        self._load_multi_modal_imgs()

        # load labels
        labels = Img.load_labels(
            dataset_dir=self._dataset_dir,
            patient=self._cur_patient,
            nii_load_func=self.__load_3d_img,
        )
        for gtv in ["gtvt", "gtvn"]:
            self._3d_imgs["{}.label".format(gtv)] = labels[gtv]

        # get slice id (after multi-modal imgs are loaded)
        self._cur_slice = self._get_middle_slice_id()

        # choose idl automatically
        # try not to reset idl id/round when patient is changed
        for gtv in ["gtvt", "gtvn"]:
            if self._idl_id[gtv] == "baseline":
                reset_id = False
            elif (
                os.path.join(self._idl_id[gtv], self._idl_round[gtv])
                not in self._combox["idl.{}".format(gtv)].currentText()
            ):
                reset_id = True
            else:
                reset_id = False
            # refresh imgs after idl.gtvn is chosen
            self.__choose_idl(gtv=gtv, reset_id=reset_id, refresh_imgs=False)

        self._refresh_imgs()
        self._refresh_title()

    def _choose_idl_gtvt(
        self, idx: int = None, reset_id: bool = True, refresh_imgs=True
    ):
        self.__choose_idl(gtv="gtvt", reset_id=reset_id, refresh_imgs=refresh_imgs)

    def _choose_idl_gtvn(
        self, idx: int = None, reset_id: bool = True, refresh_imgs=True
    ):
        self.__choose_idl(gtv="gtvn", reset_id=reset_id, refresh_imgs=refresh_imgs)

    # _choose_idl_gtvt and _choose_idl_gtvn will share this function
    def __choose_idl(self, gtv: str, reset_id: bool = True, refresh_imgs: bool = True):
        # triggered by:
        # (1) idl combox update
        # (2) patient combox update, but can not find cur patient in idl dir
        if reset_id is True:
            combox_item = self._combox["idl.{}".format(gtv)].currentText()
            if combox_item == "baseline":
                self._idl_id[gtv] = "baseline"
                self._idl_round[gtv] = "round=00"
            else:
                self._idl_id[gtv] = combox_item[: combox_item.index("/")]
                self._idl_round[gtv] = combox_item[combox_item.index("/") + 1 :]
            # self._reset_zoomin()

        # triggered by patient combox update, and find cur patient in idl.gtvn dir
        else:
            if self._idl_id[gtv] == "baseline":
                self._combox["idl.{}".format(gtv)].setCurrentText("baseline")
            else:
                self._combox["idl.{}".format(gtv)].setCurrentText(
                    os.path.join(self._idl_id[gtv], self._idl_round[gtv])
                )

        # run this after idl gtvn combox is filled
        self._enable_arrow_btns("idl.{}".format(gtv))

        # load pred (and gtvn clicks)
        if self._idl_id[gtv] == "baseline":
            pred_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "patients",
                "patient={}".format(self._cur_patient),
                "{}_pred.nii".format(gtv),
            )
            # clear gtvn clicks
            if gtv == "gtvn":
                self._3d_imgs["gtvn.clicks"] = None
        else:
            pred_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id[gtv],
                "patients",
                "patient={}".format(self._cur_patient),
                self._idl_round[gtv],
                "{}_pred.nii".format(gtv),
            )
            # load gtvn clicks
            if gtv == "gtvn":
                gtvn_clicks_path = os.path.join(
                    Path(pred_path).parent, "gtvn_clicks.nii"
                )
                self._3d_imgs["gtvn.clicks"] = self.__load_3d_img(
                    gtvn_clicks_path, binary=True
                )

        # load preds
        self._3d_imgs["{}.pred".format(gtv)] = self.__load_3d_img(
            pred_path, binary=True
        )

        # load baseline scores
        if self._idl_id[gtv] == "baseline":
            gtvn_score_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "inference_{}_{}.json".format(
                    self.__dataset_ver, self.__dataset_section
                ),
            )
            if os.path.exists(gtvn_score_path):
                gtvn_score = Json.load(gtvn_score_path)
                for metric in g.METRICS:
                    self.__scores[gtv][metric] = gtvn_score[
                        "patient={}".format(self._cur_patient)
                    ][gtv][metric]

        # load idl scores
        else:
            gtvn_score_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id[gtv],
                "inference_{}_{}.json".format(
                    self.__dataset_ver, self.__dataset_section
                ),
            )
            if os.path.exists(gtvn_score_path):
                gtvn_score = Json.load(gtvn_score_path)
                for metric in g.METRICS:
                    self.__scores[gtv][metric] = gtvn_score[
                        "patient={}".format(self._cur_patient)
                    ][metric][self._idl_round[gtv]]

        if refresh_imgs:
            self._refresh_imgs()
            self._refresh_title()

    def __choose_prev_baseline(self):
        idx = self._combox["baseline"].currentIndex() - 1
        if idx < 0:
            return
        prev_baseline = self._combox["baseline"].itemText(idx)
        self._combox["baseline"].setCurrentText(prev_baseline)
        self._choose_baseline()

    def __choose_next_baseline(self):
        idx = self._combox["baseline"].currentIndex() + 1
        if idx > self._combox["baseline"].count() - 1:
            return
        next_baseline = self._combox["baseline"].itemText(idx)
        self._combox["baseline"].setCurrentText(next_baseline)
        self._choose_baseline()

    def __choose_prev_idl_gtvn(self):
        idx = self._combox["idl.gtvn"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvn = self._combox["idl.gtvn"].itemText(idx)
        self._combox["idl.gtvn"].setCurrentText(prev_idl_gtvn)
        self._choose_idl_gtvn()

    def __choose_next_idl_gtvn(self):
        idx = self._combox["idl.gtvn"].currentIndex() + 1
        if idx > self._combox["idl.gtvn"].count() - 1:
            return
        next_idl_gtvn = self._combox["idl.gtvn"].itemText(idx)
        self._combox["idl.gtvn"].setCurrentText(next_idl_gtvn)
        self._choose_idl_gtvn()

    def __choose_prev_idl_gtvt(self):
        idx = self._combox["idl.gtvt"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvt = self._combox["idl.gtvt"].itemText(idx)
        self._combox["idl.gtvt"].setCurrentText(prev_idl_gtvt)
        self._choose_idl_gtvt()

    def __choose_next_idl_gtvt(self):
        idx = self._combox["idl.gtvt"].currentIndex() + 1
        if idx > self._combox["idl.gtvt"].count() - 1:
            return
        next_idl_gtvt = self._combox["idl.gtvt"].itemText(idx)
        self._combox["idl.gtvt"].setCurrentText(next_idl_gtvt)
        self._choose_idl_gtvt()

    def __choose_prev_patient(self):
        idx = self._combox["patient"].currentIndex() - 1
        if idx < 0:
            return
        prev_patient = self._combox["patient"].itemText(idx)
        self._combox["patient"].setCurrentText(prev_patient)
        self._choose_patient()

    def __choose_next_patient(self):
        idx = self._combox["patient"].currentIndex() + 1
        if idx > self._combox["patient"].count() - 1:
            return
        next_patient = self._combox["patient"].itemText(idx)
        self._combox["patient"].setCurrentText(next_patient)
        self._choose_patient()

    def _refresh_imgs(self):
        # no img data loaded
        if self._3d_imgs["pt"] is None:
            return

        # check if cur slice is annotated
        is_annotated = self.__is_cur_slice_annotated()

        # set contour color
        color = Dict()
        color["gtvt.label"] = self._color["gtvt.label"]
        color["gtvn.label"] = self._color["gtvn.label"]

        if is_annotated:
            color["gtvt.pred"] = self._color["gtvt.annotation"]
        else:
            color["gtvt.pred"] = self._color["gtvt.pred"]

        color["gtvn.pred"] = self._color["gtvn.pred"]
        color["gtvn.clicks"] = self._color["gtvn.annotation"]

        # load rgb imgs
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            if self.__img_plane == "sagittal":
                rgb_img = self._3d_imgs[i][:, :, self._cur_slice]
            elif self.__img_plane == "coronal":
                rgb_img = self._3d_imgs[i][:, self._cur_slice, :]
            elif self.__img_plane == "transverse":
                # for transverse plane, img is upside down,
                # true slice id is: slices_count - 1 - slice_id
                rgb_img = self._3d_imgs[i][
                    (self.__get_slices_count() - 1 - self._cur_slice), :, :
                ]
            else:
                Debug.error_exit("self.__img_plane value error")

            rgb_img = np.uint8((rgb_img - rgb_img.min()) / rgb_img.ptp() * 255.0)
            # after cv2.cvtColor, rgb_img has 3 channels, but is still numpy
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)

            # cv2.addWeighted: dst = src1 * alpha + src2 * beta + gamma
            rgb_img = cv2.addWeighted(
                src1=rgb_img,
                alpha=self.__slider["contrast.{}".format(i)].value() / 100,
                src2=np.zeros_like(rgb_img),
                beta=0,
                gamma=self.__slider["bright.{}".format(i)].value(),
            )

            # add mask to annotated slices
            selected_slices = Dict()
            total_slices_num = Dict()

            if self.__img_plane == "transverse":
                selected_slices["horizontal"] = self.__get_gtvt_selected_slices(
                    "coronal"
                )
                total_slices_num["horizontal"] = self._3d_imgs["pt"].shape[1]
                selected_slices["vertical"] = self.__get_gtvt_selected_slices(
                    "sagittal"
                )
                total_slices_num["vertical"] = self._3d_imgs["pt"].shape[2]

            elif self.__img_plane == "coronal":
                selected_slices["horizontal"] = self.__get_gtvt_selected_slices(
                    "transverse"
                )
                total_slices_num["horizontal"] = self._3d_imgs["pt"].shape[0]
                selected_slices["vertical"] = self.__get_gtvt_selected_slices(
                    "sagittal"
                )
                total_slices_num["vertical"] = self._3d_imgs["pt"].shape[2]

            elif self.__img_plane == "sagittal":
                selected_slices["horizontal"] = self.__get_gtvt_selected_slices(
                    "transverse"
                )
                total_slices_num["horizontal"] = self._3d_imgs["pt"].shape[0]
                selected_slices["vertical"] = self.__get_gtvt_selected_slices("coronal")
                total_slices_num["vertical"] = self._3d_imgs["pt"].shape[1]

            else:
                Debug.error_exit("self.__img_plane value error")

            rgb_img_zeros = np.zeros((rgb_img.shape), dtype=np.uint8)

            selected_slices_mask = None

            # annotated slices mask
            for direction in ["horizontal", "vertical"]:
                for selected_slice in selected_slices[direction]:
                    # all images are reversed in transverse plane
                    if self.__img_plane != "transverse" and direction == "horizontal":
                        slice_pos = total_slices_num[direction] - selected_slice
                    # 1mm images are reversed in sagittal plane
                    elif (
                        self.__img_plane != "sagittal"
                        and direction == "vertical"
                        and (
                            self.__dataset_ver == "au.1mm"
                            or self.__dataset_ver == "mda"
                        )
                    ):
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
                        color=self._color["gtvt.annotation"],
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
                rgb_img, self._display_frame[i]
            )
            # blur after __fit_display_frame will gain better effect
            rgb_img = cv2.GaussianBlur(rgb_img, (3, 3), cv2.BORDER_DEFAULT)

            # draw label and pred contour
            for k in [
                "gtvn.label",
                "gtvt.label",
                "gtvn.pred",
                "gtvt.pred",
                "gtvn.clicks",
            ]:
                if self._3d_imgs[k] is None:
                    continue

                # load data of current slice
                if self.__img_plane == "sagittal":
                    contours = self._3d_imgs[k][:, :, self._cur_slice].astype(np.uint8)
                elif self.__img_plane == "coronal":
                    contours = self._3d_imgs[k][:, self._cur_slice, :].astype(np.uint8)
                else:  # img_plane == "transverse":
                    # for transverse plane, img is upside down,
                    # true slice id is: slices_count - 1 - slice_id
                    contours = self._3d_imgs[k][
                        (self.__get_slices_count() - 1 - self._cur_slice), :, :
                    ].astype(np.uint8)

                contours, _ = self.__fit_display_frame(contours, self._display_frame[i])
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
            self._add_score_on_rgb_img(rgb_img)

            # add text label gtvt
            self._add_label_text_on_rgb_img(rgb_img)
            self._add_pred_text_on_rgb_img(rgb_img)

            # show imgs
            qt_image = QImage(
                rgb_img,
                rgb_img_width,
                rgb_img_height,
                rgb_img_width * rgb_img_chan,
                QImage.Format_RGB888,
            )
            self._display_frame[i].setPixmap(QPixmap.fromImage(qt_image))

    def _add_label_text_on_rgb_img(self, rgb_img):
        rgb_img_height = rgb_img.shape[0]
        pos_x = 10
        pos_y = rgb_img_height - 48

        cv_text = "LABEL:"
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["score.text"],
        )

        cv_text = "GTVt"
        pos_x += 65
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["gtvt.label"],
        )

        cv_text = "GTVn"
        pos_x += 50
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["gtvn.label"],
        )

    def _add_pred_text_on_rgb_img(self, rgb_img):
        rgb_img_height = rgb_img.shape[0]
        pos_x = 10
        pos_y = rgb_img_height - 28

        cv_text = "PRED:"
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["score.text"],
        )

        cv_text = "GTVt"
        pos_x += 65
        if self.__is_cur_slice_annotated():
            color = self._color["gtvt.annotation"]
        else:
            color = self._color["gtvt.pred"]
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=color,
        )

        # add text pred gtvn
        cv_text = "GTVn"
        pos_x += 50
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["gtvn.pred"],
        )

    def _add_score_on_rgb_img(self, rgb_img):
        text_pos_x = 10
        text_pos_y = 10
        cv_text = ""

        for metric in g.METRICS:
            # "DSC/MSD/HD95: "
            cv_text += metric.upper() + ": "
            # load scores
            for i in ["gtvt", "gtvn"]:
                if Value.is_number(self.__scores[i][metric]):
                    if metric == "dsc":
                        cv_text += "{:.2f}".format(self.__scores[i][metric])
                    elif metric == "msd":
                        cv_text += "{:.1f}".format(self.__scores[i][metric])
                    elif metric == "hd95":
                        cv_text += "{:.1f}".format(self.__scores[i][metric])
                    else:
                        Debug.error_exit("metric value error")
                else:
                    cv_text += "NaN"
                if i == "gtvt":
                    cv_text += " / "
                else:
                    cv_text += "\n"

        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(text_pos_x, text_pos_y),
            color=self._color["score.text"],
        )

    def __get_gtvt_selected_slices(self, plane) -> List:
        # get current round

        if self._idl_round["gtvt"] == "round=00":
            return []

        # load annotated slices
        idl_gtvt_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id["gtvt"]
        )
        json_path = os.path.join(
            idl_gtvt_dir,
            "patients",
            "patient={}".format(self._cur_patient),
            "selected_slices.json",
        )
        if not os.path.exists(json_path):
            return []

        selected_slices_dict = Json.load(json_path)[plane]
        selected_slices_list = List()

        for round_num in selected_slices_dict:
            selected_slices_list += List(selected_slices_dict[round_num])

            if (round_num) == self._idl_round["gtvt"]:
                break

        # change annotated slice from str to int
        for i in range(len(selected_slices_list)):
            selected_slices_list[i] = int(selected_slices_list[i])

        return selected_slices_list

    def __is_cur_slice_annotated(self) -> bool:
        if self._3d_imgs["pt"] is None:
            return False

        if int(self._cur_slice) in self.__get_gtvt_selected_slices(self.__img_plane):
            return True
        else:
            return False

    def _cv_put_text(
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
        if self._cur_slice is not None:
            slices_count = self.__get_slices_count()
            if slices_count == 0:
                return
            slice_delta = event.angleDelta().y() // 120
            if self.__img_plane == "coronal":
                slice_delta = -slice_delta
            self._cur_slice -= slice_delta
            # limite slice_id in range(0,slices_count)
            self._cur_slice %= slices_count
            self._refresh_imgs()
            self._refresh_title()

    def __get_slices_count(self) -> int:
        if (self._3d_imgs["pt"] is None) or (self.__img_plane is None):
            return 0
        elif self.__img_plane == "sagittal":
            return self._3d_imgs["pt"].shape[2]
        elif self.__img_plane == "coronal":
            return self._3d_imgs["pt"].shape[1]
        elif self.__img_plane == "transverse":
            return self._3d_imgs["pt"].shape[0]
        else:
            Debug.error_exit("self.__img_plane value error")

    def _refresh_title(self):
        win_tital = "iDL.Tool "
        # if self._idl_round["gtvt"] is not None:
        #     win_tital += "   Num.of.Annotated.Slices="
        #     win_tital += str(len(self.__get_gtvt_selected_slices(self.__img_plane)))
        if self._cur_slice is not None:
            slices_count = self.__get_slices_count()
            if slices_count > 0:
                win_tital += "   Slice={}/{}".format(self._cur_slice + 1, slices_count)
        self.setWindowTitle(win_tital)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__resize_display_frames()
        self._refresh_side_bar()
        self._refresh_imgs()

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

        self._display_frame["ct"].setGeometry(
            QRect(pos["x"][0], pos["y"][0], size["x"][0], size["y"][0])
        )
        self._display_frame["pt"].setGeometry(
            QRect(pos["x"][1], pos["y"][0], size["x"][1], size["y"][0])
        )
        self._display_frame["mrt1"].setGeometry(
            QRect(pos["x"][0], pos["y"][1], size["x"][0], size["y"][1])
        )
        self._display_frame["mrt2"].setGeometry(
            QRect(pos["x"][1], pos["y"][1], size["x"][1], size["y"][1])
        )

    def _open_file_dlg(self):
        Tk().withdraw()
        file_name = filedialog.askopenfilename()
        if file_name == "" or file_name is None:
            pass

    def _check_focus(self):
        focused_widget = QApplication.focusWidget()
        if focused_widget:
            print("Current focus:", focused_widget.objectName())
        else:
            print("No focus at the moment.")
