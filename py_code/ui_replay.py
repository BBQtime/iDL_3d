import os
import platform
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Tuple

import cv2
import numpy as np
from custom import DatasetPart, DatasetVer, Debug, Dict, DirExplorer
from custom import Global as g
from custom import Img, Json, List, Metric, Nii, Orient, Plane, Value
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPalette
from PyQt5.QtWidgets import QApplication, QButtonGroup, QMainWindow
from scipy.ndimage import measurements
from Ui_core import Ui_Core
from ui_custom_qlabel import CustomQLabel

# gravity center of gtvs


class UiReplay(QMainWindow, Ui_Core):
    def __init__(
        self,
        idl_remark: str = None,  # param: idl_remark is for subclass: UiIDL
        debug_mode: bool = False,  # param: debug_mode is for subclass: UiIDL
    ):
        super().__init__()
        self.setupUi(self)

        self._init_ui_names()
        self._init_member_var(idl_remark=idl_remark, debug_mode=debug_mode)
        self.__set_img_qlabels_background()
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
        idl_remark: str = None,  # param: idl_remark is for subclass: UiIDL
        debug_mode: bool = False,  # param: debug_mode is for subclass: UiIDL
    ):
        # load test set patients of au and mda datasets
        # DATASET_SPLIT_JSON_PATH[DatasetVer.AU_1MM] and [DatasetVer.AU_3MM] are the same
        dataset_split_au = Json.load(g.DATASET_SPLIT_JSON_PATH[DatasetVer.AU_1MM])
        dataset_split_mda = Json.load(g.DATASET_SPLIT_JSON_PATH[DatasetVer.MDA])
        self._patients = Dict()
        self._patients["au.test.inter"] = List(dataset_split_au[DatasetPart.TEST_INTER])
        self._patients["au.test.exter"] = List(dataset_split_au[DatasetPart.TEST_EXTER])
        self._patients["mda.test"] = List(dataset_split_mda[DatasetPart.TEST])

        self._baseline_id = None
        self._cur_patient = None
        self._cur_slice = 0  # starts from 0
        self._plane = None
        self._gtvs_center = None

        self._idl_id = Dict()
        self._idl_round = Dict()
        for i in ["gtvt", "gtvn"]:
            self._idl_id[i] = "baseline"
            self._idl_round[i] = "round=00"

        self._dataset_ver = None
        self._dataset_part = None
        self._nii_spacing = None  # (1,1,1) or (1,1,3)
        self._dataset_dir = None  # au.1mm / au.1mm / mda
        self.__scores = Dict()
        self._3d_imgs = Dict()
        self._rgb_img_relative_pos = None
        self.__side_bar_width = 300

        self.__gtvt_selected_slices_2d = Dict()
        self.__gtvt_selected_slices_2d[Orient.HORIZONTAL] = []
        self.__gtvt_selected_slices_2d[Orient.VERTICAL] = []

        self.__total_slices_count_2d = Dict()
        self.__total_slices_count_2d[Orient.HORIZONTAL] = 0
        self.__total_slices_count_2d[Orient.VERTICAL] = 0

    def _clear_img_data(self):
        for i in ["gtvt", "gtvn"]:
            self.__scores[i][Metric.DSC] = None
            self.__scores[i][Metric.MSD] = None
            self.__scores[i][Metric.HD95] = None

        for i in [
            "ct",
            "pt",
            "mr1",
            "mr2",
            "gtvt.label",
            "gtvn.label",
            "gtvt.pred",
            "gtvn.pred",
            "gtvt.click",
            "gtvn.clicks",
        ]:
            self._3d_imgs[i] = None

        # set image plane
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if self.__radio_btn[i].isChecked():
                self._plane = i

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
    #     for i in ["ct", "pt", "mr1", "mr2"]:
    #         left = self.img_qlabel[i].x()
    #         top = self.img_qlabel[i].y()
    #         width = self.img_qlabel[i].width()
    #         height = self.img_qlabel[i].height()
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
    #                 self._refresh_rgb_imgs()
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
    #     img_qlabel = self.img_qlabel[self.__zoomin["img"]]
    #     img_qlabel_right = img_qlabel.x() + img_qlabel.width() - 1
    #     if event.x() < img_qlabel.x():
    #         event_x = img_qlabel.x()
    #     elif event.x() > img_qlabel_right:
    #         event_x = img_qlabel_right
    #     else:
    #         event_x = event.x()
    #     img_qlabel_buttom = img_qlabel.y() + img_qlabel.height() - 1
    #     if event.y() < img_qlabel.y():
    #         event_y = img_qlabel.y()
    #     elif event.y() > img_qlabel_buttom:
    #         event_y = img_qlabel_buttom
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
    #     if self._3d_imgs["ct"] is None:
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
    #     # get img_qlabel related position
    #     img_qlabel_left = self.img_qlabel[self.__zoomin["img"]].x()
    #     img_qlabel_top = self.img_qlabel[self.__zoomin["img"]].y()
    #     start_x -= img_qlabel_left
    #     end_x -= img_qlabel_left
    #     start_y -= img_qlabel_top
    #     end_y -= img_qlabel_top

    #     # get actual_img_area related position
    #     relative_pos = self._rgb_img_relative_pos
    #     start_x -= relative_pos["x"]
    #     start_y -= relative_pos["y"]
    #     end_x -= relative_pos["x"]
    #     end_y -= relative_pos["y"]
    #     # out of range
    #     if (start_x < 0 and end_x < 0) or (
    #         start_x > relative_pos["width"] and end_x > relative_pos["width"]
    #     ):
    #         self._reset_zoomin()
    #         return
    #     if (start_y < 0 and end_y < 0) or (
    #         start_y > relative_pos["height"] and end_y > relative_pos["height"]
    #     ):
    #         self._reset_zoomin()
    #         return
    #     # limit zoomin frame in image area
    #     if start_x < 0:
    #         start_x = 0
    #     if start_y < 0:
    #         start_y = 0
    #     if end_x > relative_pos["width"]:
    #         end_x = relative_pos["width"]
    #     if end_y > relative_pos["height"]:
    #         end_y = relative_pos["height"]

    #     # get actual zoom position
    #     if self._plane == Plane.SAGITTAL:
    #         origin_width = self._3d_imgs["ct"].shape[1]
    #         origin_height = self._3d_imgs["ct"].shape[0]
    #         origin_height = round(
    #             origin_height * self._nii_spacing[2] / self._nii_spacing[1]
    #         )
    #     elif self._plane == Plane.CORONAL:
    #         origin_width = self._3d_imgs["ct"].shape[2]
    #         origin_height = self._3d_imgs["ct"].shape[0]
    #         origin_height = round(
    #             origin_height * self._nii_spacing[2] / self._nii_spacing[0]
    #         )
    #     else:
    #         origin_width = self._3d_imgs["ct"].shape[2]
    #         origin_height = self._3d_imgs["ct"].shape[1]

    #     start_x = round(start_x * origin_width / relative_pos["width"])
    #     end_x = round(end_x * origin_width / relative_pos["width"])
    #     start_y = round(start_y * origin_height / relative_pos["height"])
    #     end_y = round(end_y * origin_height / relative_pos["height"])

    #     self.__zoomin["start"] = QPoint(start_x, start_y)
    #     self.__zoomin["end"] = QPoint(end_x, end_y)
    #     self._refresh_rgb_imgs()

    def __fit_img_qlabel(self, img, img_qlabel: QtWidgets.QLabel):
        err_msg = "MainWindow.__fit_img_qlabel(), img.shape should == 2 or 3"

        # image spacing resize
        if self._plane == Plane.SAGITTAL:
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
        elif self._plane == Plane.CORONAL:
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
        relative_pos = Dict()
        relative_pos["x"], relative_pos["y"] = None, None
        relative_pos["width"], relative_pos["height"] = None, None
        final_width = img_qlabel.width()
        final_height = img_qlabel.height()

        # border on left and right
        if origin_height * final_width > final_height * origin_width:
            relative_pos["width"] = int(final_height * origin_width / origin_height)
            relative_pos["height"] = final_height
            relative_pos["x"] = int((final_width - relative_pos["width"]) / 2)
            if relative_pos["x"] < 0:
                relative_pos["x"] = 0
            relative_pos["y"] = 0
            if len(img.shape) == 3:
                black_border = np.zeros((final_height, relative_pos["x"], 3), np.uint8)
            elif len(img.shape) == 2:
                black_border = np.zeros((final_height, relative_pos["x"]), np.uint8)
            else:
                raise ValueError(err_msg)
            img = cv2.resize(
                img,
                (relative_pos["width"], relative_pos["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=1)

        # border on up and down
        else:
            relative_pos["width"] = final_width
            relative_pos["height"] = int(final_width * origin_height / origin_width)
            relative_pos["y"] = int((final_height - relative_pos["height"]) / 2)
            if relative_pos["y"] < 0:
                relative_pos["y"] = 0
            relative_pos["x"] = 0
            if len(img.shape) == 3:
                black_border = np.zeros((relative_pos["y"], final_width, 3), np.uint8)
            elif len(img.shape) == 2:
                black_border = np.zeros((relative_pos["y"], final_width), np.uint8)
            else:
                raise ValueError(err_msg)
            img = cv2.resize(
                img,
                (relative_pos["width"], relative_pos["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=0)

        # smooth img
        return img, relative_pos

    def __init_color(self):
        self._color = Dict()
        self._color["gtvt.label"] = (0, 255, 255)  # light blue
        self._color["gtvn.label"] = (0, 150, 255)  # dark blue
        self._color["gtvt.pred"] = (255, 255, 0)  # yellow
        self._color["gtvn.pred"] = (255, 128, 0)  # orange
        self._color["gtvt.annotation"] = (0, 255, 64)  # green
        self._color["gtvt.click"] = (0, 255, 64)  # green
        self._color["gtvn.clicks"] = (255, 70, 200)  # pink
        self._color["score.text"] = self._color["gtvt.annotation"]

    def _init_ui_names(self):
        self._img_qlabel_ct = CustomQLabel(self._central_widget)
        self._img_qlabel_pt = CustomQLabel(self._central_widget)
        self._img_qlabel_mr1 = CustomQLabel(self._central_widget)
        self._img_qlabel_mr2 = CustomQLabel(self._central_widget)

        self.img_qlabel = Dict()
        self.img_qlabel["ct"] = self._img_qlabel_ct
        self.img_qlabel["pt"] = self._img_qlabel_pt
        self.img_qlabel["mr1"] = self._img_qlabel_mr1
        self.img_qlabel["mr2"] = self._img_qlabel_mr2

        self._text_label = Dict()
        self._text_label["baseline"] = self._text_label_baseline
        self._text_label["patient"] = self._text_label_patient
        self._text_label["idl.gtvt"] = self._text_label_idl_gtvt
        self._text_label["idl.gtvn"] = self._text_label_idl_gtvn
        self._text_label["bright"] = self._text_label_bright
        self._text_label["contrast"] = self._text_label_contrast
        self._text_label["zoom"] = self._text_label_zoom
        self._text_label["annotation.tools"] = self._text_label_annotation_tools
        self._text_label["idl.progress"] = self._text_label_idl_progress

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
        self.__radio_btn["mr1"] = self._radio_btn_mr1
        self.__radio_btn["mr2"] = self._radio_btn_mr2
        self.__radio_btn[Plane.TRANSVERSE] = self._radio_btn_transverse
        self.__radio_btn[Plane.CORONAL] = self._radio_btn_coronal
        self.__radio_btn[Plane.SAGITTAL] = self._radio_btn_sagittal

        self.__slider = Dict()
        self.__slider["bright.ct"] = self._slider_bright_ct
        self.__slider["bright.pt"] = self._slider_bright_pt
        self.__slider["bright.mr1"] = self._slider_bright_mr1
        self.__slider["bright.mr2"] = self._slider_bright_mr2
        self.__slider["contrast.ct"] = self._slider_contrast_ct
        self.__slider["contrast.pt"] = self._slider_contrast_pt
        self.__slider["contrast.mr1"] = self._slider_contrast_mr1
        self.__slider["contrast.mr2"] = self._slider_contrast_mr2
        self.__slider["zoom"] = self._slider_zoom

    # set display frames background black
    def __set_img_qlabels_background(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, Qt.black)
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].setObjectName("")
            self.img_qlabel[i].setAutoFillBackground(True)
            self.img_qlabel[i].setPalette(pal)

    def _init_side_bar(self):
        # hide idl.gtvs controls
        self._text_label_idl_gtvs.hide()
        self._combox_idl_gtvs.hide()
        self._btn_prev_idl_gtvs.hide()
        self._btn_next_idl_gtvs.hide()

        # hide annotation controls
        self._text_box_annotation_msg.hide()
        self._text_label_annotation_tools.hide()
        self._btn_drawing_mode.hide()
        self._btn_clear.hide()
        self._btn_confirm.hide()
        self._text_label_idl_progress.hide()
        self._progress_bar_idl.hide()
        self._text_label_pen_size.hide()
        self._slider_pen_size.hide()

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
        self.__radio_btn["mr1"].setText("MR-T1")
        self.__radio_btn["mr2"].setText("MR-T2")
        self.__radio_btn[Plane.TRANSVERSE].setText("Transverse")
        self.__radio_btn[Plane.CORONAL].setText("Coronal")
        self.__radio_btn[Plane.SAGITTAL].setText("Sagittal")

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
        for i in [
            Plane.TRANSVERSE,
            Plane.CORONAL,
            Plane.SAGITTAL,
            "ct",
            "pt",
            "mr1",
            "mr2",
        ]:
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
        self.__btn_group_bright_contrast.addButton(self.__radio_btn["mr1"])
        self.__btn_group_bright_contrast.addButton(self.__radio_btn["mr2"])
        self.__btn_group_plane = QButtonGroup()
        self.__btn_group_plane.addButton(self.__radio_btn[Plane.TRANSVERSE])
        self.__btn_group_plane.addButton(self.__radio_btn[Plane.CORONAL])
        self.__btn_group_plane.addButton(self.__radio_btn[Plane.SAGITTAL])

        # radio btns checked or not
        self.__radio_btn["ct"].setChecked(True)
        self.__radio_btn["pt"].setChecked(False)
        self.__radio_btn["mr1"].setChecked(False)
        self.__radio_btn["mr2"].setChecked(False)
        self.__radio_btn[Plane.TRANSVERSE].setChecked(True)
        self.__radio_btn[Plane.CORONAL].setChecked(False)
        self.__radio_btn[Plane.SAGITTAL].setChecked(False)

        # set slider range and default value
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.__slider["bright.{}".format(i)].setMinimum(-128)
            self.__slider["bright.{}".format(i)].setMaximum(128)
            self.__slider["bright.{}".format(i)].setValue(0)
            self.__slider["contrast.{}".format(i)].setMinimum(0)
            self.__slider["contrast.{}".format(i)].setMaximum(200)
            self.__slider["contrast.{}".format(i)].setValue(100)

        # only show ct bright/contrast slider bars
        for i in ["pt", "mr1", "mr2"]:
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
            for j in ["ct", "pt", "mr1", "mr2"]:
                self.__slider["{}.{}".format(i, j)].valueChanged.connect(
                    self._refresh_rgb_imgs
                )

        self.__btn_group_plane.buttonClicked.connect(self._set_img_plane)

        self.__btn_group_bright_contrast.buttonClicked.connect(
            self.__set_bright_contrast_modality
        )

    def _fill_combox_baseline(self):
        baseline_id_list = DirExplorer.get_sub_folders(
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
            gap = 20
        else:  # windows
            gap = 40

        radio_btn_height = 25
        radio_btn_width = Dict()
        radio_btn_width["ct"] = radio_btn_width["pt"] = 45
        radio_btn_width["mr1"] = radio_btn_width["mr2"] = 60
        radio_btn_width[Plane.TRANSVERSE] = 90
        radio_btn_width[Plane.CORONAL] = 70
        radio_btn_width[Plane.SAGITTAL] = 70
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

        # brightness and contrast radio btns
        top += gap
        tmp_left = left
        for i in ["ct", "pt", "mr1", "mr2"]:
            rect = QRect(tmp_left, top, radio_btn_width[i], radio_btn_height)
            self.__radio_btn[i].setGeometry(rect)
            tmp_left += radio_btn_gap["bright.contrast"] + radio_btn_width[i]
        top += radio_btn_height

        # brightness and contrast sliders
        for i in ["bright", "contrast"]:
            rect = QRect(left, top, width, text_height)
            self._text_label[i].setGeometry(rect)
            top += text_height
            rect = QRect(left, top, width, slider_height)
            for j in ["ct", "pt", "mr1", "mr2"]:
                self.__slider["{}.{}".format(i, j)].setGeometry(rect)
            top += slider_height

        # img plane
        top += gap
        tmp_left = left
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
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

        # return the followings for UiIDL
        return left, top, width, gap, text_height, bar_height, slider_height

    def _set_img_plane(self):
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if self.__radio_btn[i].isChecked():
                self._plane = i
                break

        if self._plane == Plane.TRANSVERSE:
            self.__gtvt_selected_slices_2d[
                Orient.HORIZONTAL
            ] = self.__get_gtvt_selected_slices_on_2d(Plane.CORONAL)
            self.__total_slices_count_2d[Orient.HORIZONTAL] = self._3d_imgs["ct"].shape[
                1
            ]
            self.__gtvt_selected_slices_2d[
                Orient.VERTICAL
            ] = self.__get_gtvt_selected_slices_on_2d(Plane.SAGITTAL)
            self.__total_slices_count_2d[Orient.VERTICAL] = self._3d_imgs["ct"].shape[2]

        elif self._plane == Plane.CORONAL:
            self.__gtvt_selected_slices_2d[
                Orient.HORIZONTAL
            ] = self.__get_gtvt_selected_slices_on_2d(Plane.TRANSVERSE)
            self.__total_slices_count_2d[Orient.HORIZONTAL] = self._3d_imgs["ct"].shape[
                0
            ]
            self.__gtvt_selected_slices_2d[
                Orient.VERTICAL
            ] = self.__get_gtvt_selected_slices_on_2d(Plane.SAGITTAL)
            self.__total_slices_count_2d[Orient.VERTICAL] = self._3d_imgs["ct"].shape[2]

        elif self._plane == Plane.SAGITTAL:
            self.__gtvt_selected_slices_2d[
                Orient.HORIZONTAL
            ] = self.__get_gtvt_selected_slices_on_2d(Plane.TRANSVERSE)
            self.__total_slices_count_2d[Orient.HORIZONTAL] = self._3d_imgs["ct"].shape[
                0
            ]
            self.__gtvt_selected_slices_2d[
                Orient.VERTICAL
            ] = self.__get_gtvt_selected_slices_on_2d(Plane.CORONAL)
            self.__total_slices_count_2d[Orient.VERTICAL] = self._3d_imgs["ct"].shape[1]

        else:
            Debug.error_exit("self._plane value error")

        self._reset_cur_slice_id()
        # self._reset_zoomin()
        self._refresh_rgb_imgs()
        self._refresh_title()

    def _reset_cur_slice_id(self):
        if self._gtvs_center is not None:
            if self._plane == Plane.TRANSVERSE:
                self._cur_slice = self._gtvs_center[0]
            if self._plane == Plane.CORONAL:
                self._cur_slice = self._gtvs_center[1]
            if self._plane == Plane.SAGITTAL:
                self._cur_slice = self._gtvs_center[2]

    def __set_bright_contrast_modality(self):
        for i in ["ct", "pt", "mr1", "mr2"]:
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
        elif self.__radio_btn["mr1"].isChecked():
            key_word = "MR-T1"
        elif self.__radio_btn["mr2"].isChecked():
            key_word = "MR-T2"
        else:
            Debug.error_exit("no radio button is checked")

        self._text_label["bright"].setText("Brightness ({})".format(key_word))
        self._text_label["contrast"].setText("Contrast ({})".format(key_word))

    def _clear_img_qlabels(self):
        for i in ["ct", "pt", "mr1", "mr2"]:
            width = self.img_qlabel[i].width()
            height = self.img_qlabel[i].height()
            black_img = np.zeros([width, height, 3])
            qt_image = QImage(
                black_img,
                self.img_qlabel[i].width(),
                self.img_qlabel[i].height(),
                self.img_qlabel[i].width() * 3,
                QImage.Format_RGB888,
            )
            self.img_qlabel[i].set_background(qt_image)

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
        fold_dir = DirExplorer.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        )[0]
        baseline_dataset_ver = Json.load(os.path.join(fold_dir, "hyper.json"))[
            "dataset.ver"
        ]

        # set dataset dir based on current patient
        if self._cur_patient in self._patients["au.test.inter"]:
            self._dataset_ver = baseline_dataset_ver
            self._dataset_part = DatasetPart.TEST_INTER

        elif self._cur_patient in self._patients["au.test.exter"]:
            self._dataset_ver = baseline_dataset_ver
            self._dataset_part = DatasetPart.TEST_EXTER

        elif self._cur_patient in self._patients["mda.test"]:
            self._dataset_ver = DatasetVer.MDA
            self._dataset_part = DatasetPart.TEST
        else:
            Debug.error_exit("cant find current patient in test patients")

        # set dataset dir and nii spacing
        self._dataset_dir = g.DATASET_DIR[self._dataset_ver]
        self._nii_spacing = g.NII_SPACING[self._dataset_ver]

    def _fill_combox_patient(self):
        combox_patients = DirExplorer.get_sub_folders(
            os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, "baseline", "patients")
        )
        # from "patient=123" to "123"
        for i in range(len(combox_patients)):
            combox_patients[i] = combox_patients[i][len("patient=") :]

        combox_patients.find_identical_items(self._patients.to_list())
        combox_patients.sort()
        self._combox["patient"].addItems(combox_patients)
        self._combox["patient"].setEnabled(True)
        return combox_patients

    def _choose_baseline(self):
        # self._reset_zoomin()
        self._clear_img_data()
        self._clear_img_qlabels()

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

    def _load_3d_img(self, path: str, binary: bool = False):
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
        paths["mr1"] = "T1dr"
        paths["mr2"] = "T2dr"
        for i in ["ct", "pt", "mr1", "mr2"]:
            paths[i] = "HNCDL_{}_{}.nii".format(self._cur_patient, paths[i])
            paths[i] = os.path.join(self._dataset_dir, paths[i])
            self._3d_imgs[i] = self._load_3d_img(paths[i])

    def _get_middle_slice_id(self):
        if self._3d_imgs["ct"] is None:
            Debug.error_exit("get middle slice id after multi-modal imgs are loaded")
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
            for idl_result_dir in DirExplorer.get_sub_folders(
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
                    round_folders = DirExplorer.get_sub_folders(
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
        self.__load_labels()

        # reset slice id (after multi-modal imgs are loaded)
        # self._cur_slice = self._get_middle_slice_id()
        self._reset_cur_slice_id()

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

        self._refresh_rgb_imgs()
        self._refresh_title()

    # load labels and gtvs gravity center
    def __load_labels(self):
        labels = Img.load_labels(
            dataset_dir=self._dataset_dir,
            patient=self._cur_patient,
            nii_load_func=self._load_3d_img,
        )
        # load gtvt and gtvn
        for gtv in ["gtvt", "gtvn"]:
            self._3d_imgs["{}.label".format(gtv)] = labels[gtv]
        # load gtvs gravity center: (d,h,w)
        self._gtvs_center = list(measurements.center_of_mass(labels["gtvs"]))
        # float to int
        for i in range(len(self._gtvs_center)):
            self._gtvs_center[i] = round(self._gtvs_center[i])

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
            # clear gtvt/gtvn clicks
            if gtv == "gtvt":
                self._3d_imgs["gtvt.click"] = None
            elif gtv == "gtvn":
                self._3d_imgs["gtvn.clicks"] = None
        else:
            round_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id[gtv],
                "patients",
                "patient={}".format(self._cur_patient),
                self._idl_round[gtv],
            )
            pred_path = os.path.join(round_dir, "{}_pred.nii".format(gtv))

            # load gtvt/gtvn clicks
            if gtv == "gtvt":
                gtvt_click_path = os.path.join(round_dir, "gtvt_click.nii")
                if os.path.exists(gtvt_click_path):
                    self._3d_imgs["gtvt.click"] = self._load_3d_img(
                        gtvt_click_path, binary=True
                    )
            # load gtvt/gtvn clicks
            elif gtv == "gtvn":
                gtvn_clicks_path = os.path.join(round_dir, "gtvn_clicks.nii")
                if os.path.exists(gtvn_clicks_path):
                    self._3d_imgs["gtvn.clicks"] = self._load_3d_img(
                        gtvn_clicks_path, binary=True
                    )

        # load preds
        self._3d_imgs["{}.pred".format(gtv)] = self._load_3d_img(pred_path, binary=True)

        # load baseline scores
        if self._idl_id[gtv] == "baseline":
            gtvn_score_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "inference_{}_{}.json".format(self._dataset_ver, self._dataset_part),
            )
            if os.path.exists(gtvn_score_path):
                gtvn_score = Json.load(gtvn_score_path)
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    self.__scores[gtv][metric] = gtvn_score[
                        "patient={}".format(self._cur_patient)
                    ][gtv][metric]

        # load idl scores
        else:
            gtvn_score_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id[gtv],
                "inference_{}_{}.json".format(self._dataset_ver, self._dataset_part),
            )
            if os.path.exists(gtvn_score_path):
                gtvn_score = Json.load(gtvn_score_path)
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    self.__scores[gtv][metric] = gtvn_score[
                        "patient={}".format(self._cur_patient)
                    ][metric][self._idl_round[gtv]]

        if refresh_imgs:
            self._refresh_rgb_imgs()
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

    def _refresh_rgb_imgs(self):
        # no img data loaded
        if self._3d_imgs["ct"] is None:
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
        color["gtvt.click"] = self._color["gtvt.click"]
        color["gtvn.clicks"] = self._color["gtvn.clicks"]

        # load rgb imgs
        for i in ["ct", "pt", "mr1", "mr2"]:
            if self._plane == Plane.SAGITTAL:
                rgb_img = self._3d_imgs[i][:, :, self._cur_slice]
            elif self._plane == Plane.CORONAL:
                rgb_img = self._3d_imgs[i][:, self._cur_slice, :]
            elif self._plane == Plane.TRANSVERSE:
                rgb_img = self._3d_imgs[i][self._cur_slice, :, :]

                # # for transverse plane, img is upside down,
                # # true slice id is: slices_count - 1 - slice_id
                # rgb_img = self._3d_imgs[i][
                #     (self.__get_slices_count() - 1 - self._cur_slice), :, :
                # ]
            else:
                Debug.error_exit("self._plane value error")

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

            # add mask to gtvt selected slices
            rgb_img_zeros = np.zeros((rgb_img.shape), dtype=np.uint8)
            selected_slices_mask = None
            for direction in [Orient.HORIZONTAL, Orient.VERTICAL]:
                for gtvt_selected_slice_2d in self.__gtvt_selected_slices_2d[direction]:
                    # all images are reversed in transverse plane
                    if (
                        self._plane != Plane.TRANSVERSE
                        and direction == Orient.HORIZONTAL
                    ):
                        slice_pos = (
                            self.__total_slices_count_2d[direction]
                            - gtvt_selected_slice_2d
                        )
                    # 1mm images are reversed in sagittal plane
                    elif (
                        self._plane != Plane.SAGITTAL
                        and direction == Orient.VERTICAL
                        and (
                            self._dataset_ver == DatasetVer.AU_1MM
                            or self._dataset_ver == DatasetVer.MDA
                        )
                    ):
                        slice_pos = (
                            self.__total_slices_count_2d[direction]
                            - gtvt_selected_slice_2d
                        )
                    else:
                        slice_pos = gtvt_selected_slice_2d

                    if direction == Orient.HORIZONTAL:
                        x1 = 0
                        y1 = slice_pos
                        x2 = rgb_img.shape[1] - 1
                        y2 = slice_pos
                    elif direction == Orient.VERTICAL:
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

            # resize and fit img qlabel
            rgb_img, self._rgb_img_relative_pos = self.__fit_img_qlabel(
                rgb_img, self.img_qlabel[i]
            )

            # blur after __fit_img_qlabel will gain better effect
            rgb_img = cv2.GaussianBlur(rgb_img, (3, 3), cv2.BORDER_DEFAULT)

            # draw label and pred contour
            for k in [
                "gtvn.label",
                "gtvt.label",
                "gtvn.pred",
                "gtvt.pred",
                "gtvt.click",
                "gtvn.clicks",
            ]:
                if self._3d_imgs[k] is None:
                    continue

                # load data of current slice
                if self._plane == Plane.SAGITTAL:
                    contours = self._3d_imgs[k][:, :, self._cur_slice].astype(np.uint8)
                elif self._plane == Plane.CORONAL:
                    contours = self._3d_imgs[k][:, self._cur_slice, :].astype(np.uint8)
                elif self._plane == Plane.TRANSVERSE:
                    contours = self._3d_imgs[k][self._cur_slice, :, :].astype(np.uint8)

                    # # for transverse plane, img is upside down,
                    # # true slice id is: slices_count - 1 - slice_id
                    # contours = self._3d_imgs[k][
                    #     (self.__get_slices_count() - 1 - self._cur_slice), :, :
                    # ].astype(np.uint8)
                else:
                    Debug.error_exit("self._plane value error")

                contours, _ = self.__fit_img_qlabel(contours, self.img_qlabel[i])
                # blur after __fit_img_qlabel will make the contours looks better on the UI
                if k != "gtvn.clicks":
                    contours = cv2.GaussianBlur(contours, (7, 7), cv2.BORDER_DEFAULT)
                contours, _ = cv2.findContours(
                    contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                if k == "gtvt.click" or k == "gtvn.clicks":
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
            self.img_qlabel[i].set_background(qt_image)

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

        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            # "DSC/MSD/HD95: "
            cv_text += metric.upper() + ": "
            # load scores
            for i in ["gtvt", "gtvn"]:
                if Value.is_number(self.__scores[i][metric]):
                    if metric == Metric.DSC:
                        cv_text += "{:.2f}".format(self.__scores[i][metric])
                    elif metric == Metric.MSD:
                        cv_text += "{:.1f}".format(self.__scores[i][metric])
                    elif metric == Metric.HD95:
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

    def __get_gtvt_selected_slices_on_2d(self, img_plane) -> List:
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

        selected_slices_dict = Json.load(json_path)[img_plane]
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
        if self._3d_imgs["ct"] is None:
            return False

        if int(self._cur_slice) in self.__get_gtvt_selected_slices_on_2d(self._plane):
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
            if self._plane == Plane.CORONAL:
                slice_delta = -slice_delta
            self._cur_slice -= slice_delta
            # limite slice_id in range(0,slices_count)
            self._cur_slice %= slices_count
            self._refresh_rgb_imgs()
            self._refresh_title()

    def __get_slices_count(self) -> int:
        if (self._3d_imgs["ct"] is None) or (self._plane is None):
            return 0
        elif self._plane == Plane.SAGITTAL:
            return self._3d_imgs["ct"].shape[2]
        elif self._plane == Plane.CORONAL:
            return self._3d_imgs["ct"].shape[1]
        elif self._plane == Plane.TRANSVERSE:
            return self._3d_imgs["ct"].shape[0]
        else:
            Debug.error_exit("self._plane value error")

    def _refresh_title(self):
        win_tital = "iDL.Tool "
        # if self._idl_round["gtvt"] is not None:
        #     win_tital += "   Num.of.Annotated.Slices="
        #     win_tital += str(len(self.__get_gtvt_selected_slices_on_2d(self._plane)))
        if self._cur_slice is not None:
            slices_count = self.__get_slices_count()
            if slices_count > 0:
                win_tital += "   Slice={}/{}".format(self._cur_slice + 1, slices_count)
        self.setWindowTitle(win_tital)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__resize_img_qlabels()
        self._refresh_side_bar()
        self._refresh_rgb_imgs()

    def __resize_img_qlabels(self):
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

        self.img_qlabel["ct"].setGeometry(
            QRect(pos["x"][0], pos["y"][0], size["x"][0], size["y"][0])
        )
        self.img_qlabel["pt"].setGeometry(
            QRect(pos["x"][1], pos["y"][0], size["x"][1], size["y"][0])
        )
        self.img_qlabel["mr1"].setGeometry(
            QRect(pos["x"][0], pos["y"][1], size["x"][0], size["y"][1])
        )
        self.img_qlabel["mr2"].setGeometry(
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
