import os
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from numpy import ndarray
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import measurements
from str_lib import (
    ANNOTATION,
    AU_1MM,
    AU_3MM,
    BASELINE,
    BRIGHT,
    CLICK,
    CLICKS,
    COLOR_ENHANCE,
    CONTRAST,
    CORONAL,
    CORRECTION,
    CT,
    CT_PT_MIX,
    DATASET_VER,
    DISPLAY_MODE,
    DSC,
    GTVN,
    GTVN_CLICKS,
    GTVN_CORRECTION,
    GTVN_LABEL,
    GTVN_PRED,
    GTVN_PRED_FINAL,
    GTVS,
    GTVT,
    GTVT_ANNOTATION,
    GTVT_CLICK,
    GTVT_CORRECTION,
    GTVT_LABEL,
    GTVT_PRED,
    GTVT_PRED_FINAL,
    HD95,
    IDL_GTVN,
    IDL_GTVT,
    MDA,
    MODAL,
    MODAL_FIXED,
    MR1,
    MR2,
    MSD,
    PATIENT,
    PLANE,
    PLANE_FIXED,
    PT,
    SAGITTAL,
    TRANSVERSE,
    ZOOM,
    DatasetPart,
)
from superqt import QCollapsible
from toggle_btn import ToggleButton
from ui_custom_qlabel import CustomQLabel


class UiReplay(QMainWindow):
    def __init__(
        self,
        idl_remark: str = None,  # param: idl_remark is for subclass: UiIDL
        debug_mode: bool = False,  # param: debug_mode is for subclass: UiIDL
    ):
        super().__init__()
        self.setupUi(self)
        self._init_data(idl_remark=idl_remark, debug_mode=debug_mode)
        self._init_color()  # before init_widgets()
        self._init_widgets()  # after _init_data() and init_widgets()
        self.setMinimumSize(self.__side_bar_width + 600, 600)  # after _init_data()
        # self.__init_zoomin()
        self.resize(1200, 800)  # set origin size
        self.showMaximized()
        self._load_baseline_data()  # load first baseline result

    def _init_data(
        self,
        idl_remark: str = None,  # param: idl_remark is for subclass: UiIDL
        debug_mode: bool = False,  # param: debug_mode is for subclass: UiIDL
    ):
        # load test set patients of au and mda datasets
        # DATASET_SPLIT_JSON_PATH[AU_1MM] and [AU_3MM] are the same
        dataset_split_au = Json.load(g.DATASET_SPLIT_JSON_PATH[AU_1MM])
        dataset_split_mda = Json.load(g.DATASET_SPLIT_JSON_PATH[MDA])
        self._patients = Dict()
        self._patients["au.test.inter"] = List(dataset_split_au[DatasetPart.TEST_INTER])
        self._patients["au.test.exter"] = List(dataset_split_au[DatasetPart.TEST_EXTER])
        self._patients["mda.test"] = List(dataset_split_mda[DatasetPart.TEST])

        self._baseline_id = None
        self._cur_patient = None
        self.cur_slice_id = Dict()
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            self.cur_slice_id[i] = 0  # starts from 0
        self._gtvs_center = None

        self._idl_id = Dict()
        self._idl_round = Dict()
        for i in [GTVT, GTVN]:
            self._idl_id[i] = BASELINE
            self._idl_round[i] = "round=00"

        self._dataset_ver = None
        self._dataset_part = None
        self._nii_spacing = None  # (1,1,1) or (1,1,3)
        self._dataset_dir = None  # au.1mm / au.1mm / mda

        self.__scores = Dict()
        self.__clear_scores()
        self.img_3d = Dict()
        self._clear_img_3d()

        self._rgb_img_roi = None
        self.__side_bar_width = 310

        self.__clear_gtvt_selected_slices_3d()

        # self.__gtvt_selected_slices_2d = Dict()
        # self.__gtvt_selected_slices_2d[Orient.HORIZONTAL] = List()
        # self.__gtvt_selected_slices_2d[Orient.VERTICAL] = List()

        # self.__total_slices_count_2d = Dict()
        # self.__total_slices_count_2d[Orient.HORIZONTAL] = 0
        # self.__total_slices_count_2d[Orient.VERTICAL] = 0

    def _clear_img_3d(self):
        for i in [
            CT,
            PT,
            MR1,
            MR2,
            GTVT_LABEL,
            GTVN_LABEL,
            GTVT_PRED,
            GTVN_PRED,
            GTVT_CLICK,
            GTVN_CLICKS,
            GTVT_ANNOTATION,
            GTVT_CORRECTION,
            GTVN_CORRECTION,
            GTVT_PRED_FINAL,
            GTVN_PRED_FINAL,
        ]:
            self.img_3d[i] = None

    def __clear_scores(self):
        for i in [GTVT, GTVN]:
            self.__scores[i][DSC] = None
            self.__scores[i][MSD] = None
            self.__scores[i][HD95] = None

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
    #     for i in [CT, PT, MR1, MR2]:
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
    #                 self.refresh_img_qlabels()
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
    #     if self.img_3d[CT] is None:
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
    #     rgb_img_roi = self._rgb_img_roi
    #     start_x -= rgb_img_roi["x"]
    #     start_y -= rgb_img_roi["y"]
    #     end_x -= rgb_img_roi["x"]
    #     end_y -= rgb_img_roi["y"]
    #     # out of range
    #     if (start_x < 0 and end_x < 0) or (
    #         start_x > rgb_img_roi["width"] and end_x > rgb_img_roi["width"]
    #     ):
    #         self._reset_zoomin()
    #         return
    #     if (start_y < 0 and end_y < 0) or (
    #         start_y > rgb_img_roi["height"] and end_y > rgb_img_roi["height"]
    #     ):
    #         self._reset_zoomin()
    #         return
    #     # limit zoomin frame in image area
    #     if start_x < 0:
    #         start_x = 0
    #     if start_y < 0:
    #         start_y = 0
    #     if end_x > rgb_img_roi["width"]:
    #         end_x = rgb_img_roi["width"]
    #     if end_y > rgb_img_roi["height"]:
    #         end_y = rgb_img_roi["height"]

    #     # get actual zoom position
    #     if self._plane == SAGITTAL:
    #         origin_width = self.img_3d[CT].shape[1]
    #         origin_height = self.img_3d[CT].shape[0]
    #         origin_height = round(
    #             origin_height * self._nii_spacing[2] / self._nii_spacing[1]
    #         )
    #     elif self._plane == CORONAL:
    #         origin_width = self.img_3d[CT].shape[2]
    #         origin_height = self.img_3d[CT].shape[0]
    #         origin_height = round(
    #             origin_height * self._nii_spacing[2] / self._nii_spacing[0]
    #         )
    #     else:
    #         origin_width = self.img_3d[CT].shape[2]
    #         origin_height = self.img_3d[CT].shape[1]

    #     start_x = round(start_x * origin_width / rgb_img_roi["width"])
    #     end_x = round(end_x * origin_width / rgb_img_roi["width"])
    #     start_y = round(start_y * origin_height / rgb_img_roi["height"])
    #     end_y = round(end_y * origin_height / rgb_img_roi["height"])

    #     self.__zoomin["start"] = QPoint(start_x, start_y)
    #     self.__zoomin["end"] = QPoint(end_x, end_y)
    #     self.refresh_img_qlabels()

    def _fit_img_qlabel(self, img, img_qlabel: QLabel):
        err_msg = "MainWindow._fit_img_qlabel(), img.shape should == 2 or 3"

        # spacing upscalling
        if self._nii_spacing[2] != 1.0 and img_qlabel.plane == SAGITTAL:
            spacing_height = round(
                img.shape[0] * self._nii_spacing[2] / self._nii_spacing[1]
            )
            img = cv2.resize(
                img,
                (
                    img.shape[1],
                    spacing_height,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        elif self._nii_spacing[2] != 1.0 and img_qlabel.plane == CORONAL:
            spacing_height = round(
                img.shape[0] * self._nii_spacing[2] / self._nii_spacing[0]
            )
            img = cv2.resize(
                img,
                (
                    img.shape[1],
                    spacing_height,
                ),
                interpolation=cv2.INTER_CUBIC,
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
        rgb_img_roi = Dict()
        rgb_img_roi["x"], rgb_img_roi["y"] = None, None
        rgb_img_roi["width"], rgb_img_roi["height"] = None, None
        final_width = img_qlabel.width()
        final_height = img_qlabel.height()

        # border on left and right
        if origin_height * final_width > final_height * origin_width:
            rgb_img_roi["width"] = int(final_height * origin_width / origin_height)
            rgb_img_roi["height"] = final_height
            rgb_img_roi["x"] = int((final_width - rgb_img_roi["width"]) / 2)
            if rgb_img_roi["x"] < 0:
                rgb_img_roi["x"] = 0
            rgb_img_roi["y"] = 0
            if len(img.shape) == 3:
                black_border = np.zeros((final_height, rgb_img_roi["x"], 3), np.uint8)
            elif len(img.shape) == 2:
                black_border = np.zeros((final_height, rgb_img_roi["x"]), np.uint8)
            else:
                raise ValueError(err_msg)
            img = cv2.resize(
                img,
                (rgb_img_roi["width"], rgb_img_roi["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=1)

        # border on up and down
        else:
            rgb_img_roi["width"] = final_width
            rgb_img_roi["height"] = int(final_width * origin_height / origin_width)
            rgb_img_roi["y"] = int((final_height - rgb_img_roi["height"]) / 2)
            if rgb_img_roi["y"] < 0:
                rgb_img_roi["y"] = 0
            rgb_img_roi["x"] = 0
            if len(img.shape) == 3:
                black_border = np.zeros((rgb_img_roi["y"], final_width, 3), np.uint8)
            elif len(img.shape) == 2:
                black_border = np.zeros((rgb_img_roi["y"], final_width), np.uint8)
            else:
                raise ValueError(err_msg)
            img = cv2.resize(
                img,
                (rgb_img_roi["width"], rgb_img_roi["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=0)

        # smooth img
        return img, rgb_img_roi

    def _init_color(self):
        self._color = Dict()
        self._color["black"] = (0, 0, 0)
        self._color["red"] = (255, 50, 0)
        self._color["green"] = (0, 255, 64)
        self._color["magenta"] = (255, 70, 200)
        self._color["cyan"] = (0, 255, 255)
        self._color["blue"] = (0, 160, 255)
        self._color["yellow"] = (255, 255, 0)
        self._color["orange"] = (255, 120, 0)
        self._color[GTVT_PRED] = self._color["yellow"]
        self._color[GTVT_LABEL] = self._color["orange"]
        self._color[GTVN_PRED] = self._color["cyan"]
        self._color[GTVN_LABEL] = self._color["blue"]
        self._color[GTVT_ANNOTATION] = self._color["magenta"]
        self._color[GTVT_CORRECTION] = self._color["red"]
        self._color[GTVN_CORRECTION] = self._color["red"]
        self._color[GTVT_CLICK] = self._color["magenta"]
        self._color[GTVN_CLICKS] = self._color["magenta"]

    def setupUi(self, Core):
        Core.setObjectName("Core")
        self._central_widget = QtWidgets.QWidget(Core)
        # self._central_widget.setObjectName("_central_widget")
        Core.setCentralWidget(self._central_widget)
        self.retranslateUi(Core)
        QtCore.QMetaObject.connectSlotsByName(Core)

    def retranslateUi(self, Core):
        _translate = QtCore.QCoreApplication.translate
        Core.setWindowTitle(_translate("Core", "Interactive Learning Tool"))

    def _init_widgets_img_qlabels(self):
        self.img_qlabel = Dict()
        pal = QPalette()
        pal.setColor(QPalette.Window, Qt.black)

        for i in [
            CT,
            PT,
            MR1,
            MR2,
            TRANSVERSE,
            CORONAL,
            SAGITTAL,
        ]:
            self.img_qlabel[i] = CustomQLabel(self._central_widget)
            self.img_qlabel[i].setObjectName("")
            # black background
            self.img_qlabel[i].setAutoFillBackground(True)
            self.img_qlabel[i].setPalette(pal)

        # fixed plane
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            self.img_qlabel[i].plane = i

        # fixed modal
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].modal = i

        self.img_qlabel[TRANSVERSE].modal = CT_PT_MIX

    def _init_widgets_combox(self):
        self._combox = Dict()
        for i in [BASELINE, PATIENT, IDL_GTVT, IDL_GTVN]:
            self._combox[i] = QtWidgets.QComboBox(self._central_widget)
            self._combox[i].setFixedHeight(30)
            # set combobox dropdown width: 700px
            if i != PATIENT:
                self._combox[i].setStyleSheet(
                    """*
                    QComboBox QAbstractItemView
                    {
                        min-width: 500px;
                    }
                    """
                )
            if i in [PATIENT, IDL_GTVT, IDL_GTVN]:
                self._combox[i].setEnabled(False)

        # fill combox baseline
        baseline_id_list = Dir.get_sub_dirs(
            g.TRAIN_RESULTS_DIR, key_word="baseline_", shuffle=False
        )
        self._combox[BASELINE].addItems(baseline_id_list)

        # set real idl baseline id as default
        for baseline_id in baseline_id_list:
            if "real.idl" in baseline_id:
                real_idl_baseline_id = baseline_id
        self._combox[BASELINE].setCurrentText(real_idl_baseline_id)

        # arrow buttons
        self._arrow_btn = Dict()
        for i in ["prev", "next"]:
            for j in [BASELINE, PATIENT, IDL_GTVT, IDL_GTVN]:
                self._arrow_btn["{}.{}".format(i, j)] = QtWidgets.QToolButton()
                self._arrow_btn["{}.{}".format(i, j)].setFixedWidth(30)
                self._arrow_btn["{}.{}".format(i, j)].setFixedHeight(30)

        # set arrow buttons initial state
        for i in [BASELINE, PATIENT, IDL_GTVT, IDL_GTVN]:
            self._arrow_btn["prev.{}".format(i)].setArrowType(Qt.LeftArrow)
            self._arrow_btn["next.{}".format(i)].setArrowType(Qt.RightArrow)
            if i in [PATIENT, IDL_GTVT, IDL_GTVN]:
                self._arrow_btn["prev.{}".format(i)].setEnabled(False)
                self._arrow_btn["next.{}".format(i)].setEnabled(False)

        # collapse - baseline/patient/idl.gtvt/gtvn
        self._collap[BASELINE] = QCollapsible("SELECT BASELINE")
        self._collap[PATIENT] = QCollapsible("SELECT PATIENT")
        self._collap[IDL_GTVT] = QCollapsible("SELECT IDL GTVT")
        self._collap[IDL_GTVN] = QCollapsible("SELECT IDL GTVN")
        for i in [BASELINE, PATIENT, IDL_GTVT, IDL_GTVN]:
            # self._collap[i].setFixedHeight(90)
            self._collap[i].expand(True)
            h_layout = QHBoxLayout()
            h_layout.setSpacing(1)
            h_layout.addWidget(self._arrow_btn["prev.{}".format(i)])
            h_layout.addWidget(self._combox[i])
            h_layout.addWidget(self._arrow_btn["next.{}".format(i)])
            container = QWidget()
            container.setLayout(h_layout)
            self._collap[i].addWidget(container)

        # connect ctrls to functions
        self._combox[BASELINE].activated.connect(self._load_baseline_data)
        self._arrow_btn["prev.baseline"].clicked.connect(self._load_prev_baseline_data)
        self._arrow_btn["next.baseline"].clicked.connect(self._load_next_baseline_data)
        self._combox[PATIENT].activated.connect(self._load_patient_data)
        self._arrow_btn["prev.patient"].clicked.connect(self._load_prev_patient_data)
        self._arrow_btn["next.patient"].clicked.connect(self._load_next_patient_data)
        self._combox[IDL_GTVT].activated.connect(self._load_idl_gtvt_data)
        self._arrow_btn["prev.idl.gtvt"].clicked.connect(self._load_prev_idl_gtvt_data)
        self._arrow_btn["next.idl.gtvt"].clicked.connect(self._load_next_idl_gtvt_data)
        self._combox[IDL_GTVN].activated.connect(self._load_idl_gtvn_data)
        self._arrow_btn["prev.idl.gtvn"].clicked.connect(self._load_prev_idl_gtvn_data)
        self._arrow_btn["next.idl.gtvn"].clicked.connect(self._load_next_idl_gtvn_data)

    def _init_widgets_color_enhance(self):
        # init radio btns
        for i in [MODAL, PLANE]:
            self._radio_group[COLOR_ENHANCE][i] = QButtonGroup()

        for i in [CT, PT, MR1, MR2]:
            self._radio_btn[COLOR_ENHANCE][i] = QRadioButton()
            self._radio_group[COLOR_ENHANCE][MODAL].addButton(
                self._radio_btn[COLOR_ENHANCE][i]
            )

        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            self._radio_btn[COLOR_ENHANCE][i] = QRadioButton()
            self._radio_group[COLOR_ENHANCE][PLANE].addButton(
                self._radio_btn[COLOR_ENHANCE][i]
            )

        # set text modal
        self._radio_btn[COLOR_ENHANCE][CT].setText("CT")
        self._radio_btn[COLOR_ENHANCE][PT].setText("PT")
        self._radio_btn[COLOR_ENHANCE][MR1].setText("MR-T1")
        self._radio_btn[COLOR_ENHANCE][MR2].setText("MR-T2")
        # set text plane
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            self._radio_btn[COLOR_ENHANCE][i].setText(i.capitalize())

        # set checked
        self._radio_btn[COLOR_ENHANCE][CT].setChecked(True)
        self._radio_btn[COLOR_ENHANCE][TRANSVERSE].setChecked(True)

        # text labels
        for i in [BRIGHT, CONTRAST]:
            self._text_label[i] = QLabel()
        self._text_label[BRIGHT].setText("Brightness (CT)")
        self._text_label[CONTRAST].setText("Contrast (CT)")

        # slider bars
        for i in [BRIGHT, CONTRAST]:
            for j in [CT, PT, MR1, MR2, TRANSVERSE, CORONAL, SAGITTAL]:
                self._slider[i][j] = QSlider()
                slider = self._slider[i][j]
                slider.setOrientation(Qt.Horizontal)
                if i == BRIGHT:
                    slider.setMinimum(-128)
                    slider.setMaximum(128)
                    slider.setValue(0)
                elif i == CONTRAST:
                    slider.setMinimum(0)
                    slider.setMaximum(200)
                    slider.setValue(100)
                # only show ct slider bars
                if j != CT:
                    slider.hide()

        # collapse
        self._collap[COLOR_ENHANCE] = QCollapsible("COLOR ENHANCEMENT")
        # self._collap[COLOR_ENHANCE].setFixedHeight(180)
        self._collap[COLOR_ENHANCE].expand(False)
        v_layout = QVBoxLayout()

        # radio buttons: ct/pt/mr1/mr2
        h_layout = QHBoxLayout()
        for i in [CT, PT, MR1, MR2]:
            h_layout.addWidget(self._radio_btn[COLOR_ENHANCE][i])
        v_layout.addLayout(h_layout)

        # radio buttons: transverse/coronal/sagittal
        h_layout = QHBoxLayout()
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            h_layout.addWidget(self._radio_btn[COLOR_ENHANCE][i])
        v_layout.addLayout(h_layout)

        # text labels and slider bars
        for i in [BRIGHT, CONTRAST]:
            v_layout.addWidget(self._text_label[i])
            for j in [CT, PT, MR1, MR2, TRANSVERSE, CORONAL, SAGITTAL]:
                v_layout.addWidget(self._slider[i][j])

        # add final layout into collapsible space
        container = QWidget()
        container.setLayout(v_layout)
        self._collap[COLOR_ENHANCE].addWidget(container)

        # connect widgets to functions
        for i in [BRIGHT, CONTRAST]:
            for j in [CT, PT, MR1, MR2, TRANSVERSE, CORONAL, SAGITTAL]:
                self._slider[i][j].valueChanged.connect(self.refresh_img_qlabels)
        for i in [MODAL, PLANE]:
            self._radio_group[COLOR_ENHANCE][i].buttonClicked.connect(
                self.__switch_color_enhance_slider_bars
            )

    def _plane_fixed_mode_switch_coronal_modal(self):
        for modal in self._radio_btn[PLANE_FIXED][CORONAL].keys():
            if self._radio_btn[PLANE_FIXED][CORONAL][modal].isChecked():
                self.img_qlabel[CORONAL].modal = modal
                break
        self.refresh_img_qlabels(img_name=CORONAL)

    def _plane_fixed_mode_switch_sagittal_modal(self):
        for modal in self._radio_btn[PLANE_FIXED][SAGITTAL].keys():
            if self._radio_btn[PLANE_FIXED][SAGITTAL][modal].isChecked():
                self.img_qlabel[SAGITTAL].modal = modal
                break
        self.refresh_img_qlabels(img_name=SAGITTAL)

    def display_mode(self):
        if self._toggle_btn.isChecked():
            return PLANE_FIXED
        else:
            return MODAL_FIXED

    def switch_display_mode(self):
        display_mode = self.display_mode()

        # img qlabels: modalities
        for i in [CT, PT, MR1, MR2]:
            if display_mode == PLANE_FIXED:
                self.img_qlabel[i].hide()
            else:
                self.img_qlabel[i].show()

        # img qlabels: planes
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            if display_mode == PLANE_FIXED:
                self.img_qlabel[i].show()
            else:
                self.img_qlabel[i].hide()

        # plane fixed mode: text labels
        for i in [CT, PT, TRANSVERSE, CORONAL, SAGITTAL]:
            if display_mode == PLANE_FIXED:
                self._text_label[i].show()
            else:
                self._text_label[i].hide()

        # radio buttons: modal fixed mode
        for i in self._radio_btn[MODAL_FIXED].keys():
            if display_mode == PLANE_FIXED:
                self._radio_btn[MODAL_FIXED][i].hide()
            else:
                self._radio_btn[MODAL_FIXED][i].show()

        # ratio buttons: plane fixed mode
        for i in [CORONAL, SAGITTAL]:
            for j in [PT, MR1, MR2]:
                if display_mode == PLANE_FIXED:
                    self._radio_btn[PLANE_FIXED][i][j].show()
                else:
                    self._radio_btn[PLANE_FIXED][i][j].hide()

        # color enhancement radio buttons and slider bars
        if display_mode == PLANE_FIXED:
            show_list = [TRANSVERSE, CORONAL, SAGITTAL]
            hide_list = [CT, PT, MR1, MR2]
        else:
            show_list = [CT, PT, MR1, MR2]
            hide_list = [TRANSVERSE, CORONAL, SAGITTAL]

        # hide radio buttons
        for i in hide_list:
            self._radio_btn[COLOR_ENHANCE][i].hide()
            # # hide slider bars
            # for j in [BRIGHT, CONTRAST]:
            #     self._slider[j][i].hide()

        # show radio buttons
        for i in show_list:
            self._radio_btn[COLOR_ENHANCE][i].show()
            # # show slider bars
            # if self._radio_btn[COLOR_ENHANCE][i].isChecked():
            #     for j in [BRIGHT, CONTRAST]:
            #         self._slider[j][i].show()

        self.__switch_color_enhance_slider_bars()

        # ct/pt mix slider
        if display_mode == PLANE_FIXED:
            self._slider[CT_PT_MIX].show()
        else:
            self._slider[CT_PT_MIX].hide()

        self.refresh_img_qlabels()

    def _init_widgets_display_mode(self):
        # toggle display mode
        for i in [MODAL_FIXED, PLANE_FIXED]:
            self._text_label[i] = QLabel()
            # self._text_label[i].setStyleSheet("border: 1px solid black;")
        self._text_label[MODAL_FIXED].setText("Modality Fixed")
        self._text_label[MODAL_FIXED].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._text_label[MODAL_FIXED].setFixedWidth(112)
        self._text_label[PLANE_FIXED].setText("Plane Fixed")
        self._text_label[PLANE_FIXED].setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        #  display mode: modality fixed
        self._radio_group[MODAL_FIXED] = QButtonGroup()
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            self._radio_btn[MODAL_FIXED][i] = QRadioButton()
            self._radio_btn[MODAL_FIXED][i].setText(i.capitalize())
            self._radio_group[MODAL_FIXED].addButton(self._radio_btn[MODAL_FIXED][i])

        self._radio_btn[MODAL_FIXED][TRANSVERSE].setFixedWidth(120)
        # set checked
        self._radio_btn[MODAL_FIXED][TRANSVERSE].setChecked(True)
        # connect ui to functions
        self._radio_group[MODAL_FIXED].buttonClicked.connect(
            self._modal_fixed_mode_switch_plane
        )

        # display mode: plane fixed
        # radio buttons
        for i in [CORONAL, SAGITTAL]:
            self._radio_group[PLANE_FIXED][i] = QButtonGroup()
            for j in [PT, MR1, MR2]:
                self._radio_btn[PLANE_FIXED][i][j] = QRadioButton()
                self._radio_group[PLANE_FIXED][i].addButton(
                    self._radio_btn[PLANE_FIXED][i][j]
                )
        # connect functions
        self._radio_group[PLANE_FIXED][CORONAL].buttonClicked.connect(
            self._plane_fixed_mode_switch_coronal_modal
        )
        self._radio_group[PLANE_FIXED][SAGITTAL].buttonClicked.connect(
            self._plane_fixed_mode_switch_sagittal_modal
        )

        for i in [CORONAL, SAGITTAL]:
            # set checked
            self._radio_btn[PLANE_FIXED][i][MR1].setChecked(True)
            # set text
            self._radio_btn[PLANE_FIXED][i][PT].setText("PT")
            self._radio_btn[PLANE_FIXED][i][MR1].setText("MR-T1")
            self._radio_btn[PLANE_FIXED][i][MR2].setText("MR-T2")

        # reset image plane
        for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
            if self._radio_btn[MODAL_FIXED][plane].isChecked():
                for modal in [CT, PT, MR1, MR2]:
                    self.img_qlabel[modal].plane = plane
                break

        # reset img modality
        for plane in [CORONAL, SAGITTAL]:
            for modal in [PT, MR1, MR2]:
                if self._radio_btn[PLANE_FIXED][plane][modal].isChecked():
                    self.img_qlabel[plane].modal = modal
                    continue

        # text label for plane fixed mode
        for i in [CT, PT]:
            self._text_label[i] = QLabel()
            self._text_label[i].setText(i.upper())
            # self._text_label[i].setStyleSheet("border: 1px solid black;")
        self._text_label[CT].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._text_label[PT].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._text_label[CT].setFixedWidth(30)

        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            self._text_label[i] = QLabel()
            self._text_label[i].setText(i.capitalize())
            if i != TRANSVERSE:
                self._text_label[i].setFixedWidth(63)
        # self._text_label[TRANSVERSE].setStyleSheet("border: 1px solid black;")

        # ct/pt weight slider bar
        self._slider[CT_PT_MIX] = QSlider()
        self._slider[CT_PT_MIX].setOrientation(Qt.Horizontal)
        self._slider[CT_PT_MIX].setMinimum(0)
        self._slider[CT_PT_MIX].setMaximum(100)
        self._slider[CT_PT_MIX].setValue(50)
        self._slider[CT_PT_MIX].setFixedWidth(112)
        self._slider[CT_PT_MIX].valueChanged.connect(
            self.__refresh_img_qlabel_trasverse
        )

        # toggle button
        self._toggle_btn = ToggleButton(is_checked=0)

        # collapse
        self._collap[DISPLAY_MODE] = QCollapsible("DISPLAY MODE")
        # self._collap[DISPLAY_MODE].setFixedHeight(170)
        self._collap[DISPLAY_MODE].expand(True)
        v_layout = QVBoxLayout()

        # toggle btn and display mode text
        h_layout = QHBoxLayout()
        h_layout.addWidget(self._text_label[MODAL_FIXED])
        h_layout.addWidget(self._toggle_btn)
        h_layout.addWidget(self._text_label[PLANE_FIXED])
        v_layout.addLayout(h_layout)

        # modality fixed widgets
        h_layout = QHBoxLayout()
        for i in [TRANSVERSE, CORONAL, SAGITTAL]:
            h_layout.addWidget(self._radio_btn[MODAL_FIXED][i])
        v_layout.addLayout(h_layout)

        # plane fixed widgets - transverse
        h_layout = QHBoxLayout()
        h_layout.addWidget(self._text_label[TRANSVERSE])
        h_layout.addWidget(self._text_label[CT])
        h_layout.addWidget(self._slider[CT_PT_MIX])
        h_layout.addWidget(self._text_label[PT])
        v_layout.addLayout(h_layout)

        # plane fixed widgets
        for i in [CORONAL, SAGITTAL]:
            h_layout = QHBoxLayout()
            h_layout.addWidget(self._text_label[i])
            for j in [PT, MR1, MR2]:
                h_layout.addWidget(self._radio_btn[PLANE_FIXED][i][j])
            v_layout.addLayout(h_layout)

        # put v_layout into collapsible space
        container = QWidget()
        container.setLayout(v_layout)
        self._collap[DISPLAY_MODE].addWidget(container)

    def _init_widgets_set_fonts(self):
        for i in self._collap.keys():
            self._collap[i].setStyleSheet("font-weight: bold; color: white;")

        self._font_bold = QFont("Arial", 10)
        self._font_light = QFont("Arial", 10)
        self._font_bold.setBold(True)
        self._font_light.setBold(False)

        for i in self._text_label.keys():
            self._text_label[i].setFont(self._font_bold)
            self._text_label[i].setStyleSheet("color: white;")

        for i in self._collap.keys():
            self._collap[i].setFont(self._font_bold)

        for i in self._combox.keys():
            self._combox[i].setFont(self._font_bold)

        for i in self._radio_btn[COLOR_ENHANCE].keys():
            self._radio_btn[COLOR_ENHANCE][i].setFont(self._font_bold)

        for i in self._radio_btn[PLANE_FIXED].keys():
            for j in self._radio_btn[PLANE_FIXED][i].keys():
                self._radio_btn[PLANE_FIXED][i][j].setFont(self._font_bold)

        for i in self._radio_btn[MODAL_FIXED].keys():
            self._radio_btn[MODAL_FIXED][i].setFont(self._font_bold)

    def _init_widgets_zoom(self):
        self._slider[ZOOM] = QSlider()
        self._slider[ZOOM].setOrientation(Qt.Horizontal)
        self._slider[ZOOM].setMinimum(100)
        self._slider[ZOOM].setMaximum(200)
        self._slider[ZOOM].setValue(100)
        # add slider into collapsible space
        self._collap[ZOOM] = QCollapsible("ZOOM IN")
        # self._collap[ZOOM].setFixedHeight(60)
        self._collap[ZOOM].addWidget(self._slider[ZOOM])

    # virtual function (for ui_idl)
    def _init_widgets_annotation(self):
        return

    def _init_widgets(self):
        self._collap = Dict()
        self._radio_btn = Dict()
        self._radio_group = Dict()
        self._text_label = Dict()
        self._slider = Dict()

        self._init_widgets_combox()
        self._init_widgets_img_qlabels()
        self._init_widgets_annotation()
        self._init_widgets_display_mode()
        self._init_widgets_color_enhance()
        self._init_widgets_zoom()
        self._init_widgets_set_fonts()
        self.switch_display_mode()  # after all widgets initalized

        # add collapsible bars into sidebar
        v_layout = QVBoxLayout()
        v_layout.setSpacing(1)
        for i in self._collap.keys():
            v_layout.addWidget(self._collap[i])
        self._side_bar = QWidget(self._central_widget)
        self._side_bar.setLayout(v_layout)

    def _refresh_side_bar(self):
        left = 0
        top = 0
        width = self.__side_bar_width - left * 2
        height = self._central_widget.height()
        left += self.geometry().width() - self.__side_bar_width
        rect = QRect(left, top, width, height)
        self._side_bar.setGeometry(rect)
        return

        # text_height = 25
        # bar_height = 25
        # slider_height = 20
        # arrow_btn_width = 30

        # if platform.system().lower() == "linux":
        #     gap = 30
        # else:  # windows
        #     gap = 40

        # radio_btn_height = 25
        # radio_btn_width = Dict()
        # radio_btn_width[CT] = radio_btn_width[PT] = 45
        # radio_btn_width[MR1] = radio_btn_width[MR2] = 60
        # radio_btn_width[TRANSVERSE] = 90
        # radio_btn_width[CORONAL] = 70
        # radio_btn_width[SAGITTAL] = 70
        # radio_btn_gap = Dict()
        # radio_btn_gap["luminance"] = 10
        # radio_btn_gap["planes"] = 6

        # # side bar location
        # side_bar_x = self.geometry().width() - self.__side_bar_width
        # width = self.__side_bar_width - left * 2
        # left += side_bar_x

        # # set position of text label / comboxes / btns
        # for i in widgets_to_display:
        #     # text label
        #     top += gap
        #     rect = QRect(left, top, width, text_height)
        #     self._text_label[i].setGeometry(rect)
        #     top += text_height

        #     # btn prev
        #     tmp_left = left
        #     rect = QRect(tmp_left, top, arrow_btn_width, bar_height)
        #     self._arrow_btn["prev.{}".format(i)].setGeometry(rect)

        #     # combobox
        #     tmp_left += arrow_btn_width
        #     rect = QRect(tmp_left + 1, top, width - arrow_btn_width * 2 - 2, bar_height)
        #     self._combox[i].setGeometry(rect)

        #     # btn next
        #     tmp_left += width - arrow_btn_width * 2
        #     rect = QRect(tmp_left, top, arrow_btn_width, bar_height)
        #     self._arrow_btn["next.{}".format(i)].setGeometry(rect)

        #     # next element
        #     top += bar_height

        # # return the followings for UiIDL
        # return (
        #     left,
        #     top,
        #     width,
        #     gap,
        #     text_height,
        #     bar_height,
        #     slider_height,
        #     radio_btn_height,
        # )

    # new_plane = None will read from radio buttons
    def _modal_fixed_mode_switch_plane(
        self, connected_radio_btn: QRadioButton = None, new_plane: str = None
    ):
        if new_plane is None:
            for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
                if self._radio_btn[MODAL_FIXED][plane].isChecked():
                    for modal in [CT, PT, MR1, MR2]:
                        self.img_qlabel[modal].plane = plane
                    break

        else:
            for modal in [CT, PT, MR1, MR2]:
                self.img_qlabel[modal].plane = new_plane
            for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
                if plane == new_plane:
                    self._radio_btn[MODAL_FIXED][plane].setChecked(True)
                else:
                    self._radio_btn[MODAL_FIXED][plane].setChecked(False)

        # self.__refresh_gtvt_selected_slices_2d()
        # self._reset_zoomin()
        self.refresh_img_qlabels()

    # def __refresh_gtvt_selected_slices_2d(self):
    #     if self.img_3d[CT] is None:
    #         return

    #     if self._plane == TRANSVERSE:
    #         self.__gtvt_selected_slices_2d[
    #             Orient.HORIZONTAL
    #         ] = self.__gtvt_selected_slices_3d[CORONAL]
    #         self.__gtvt_selected_slices_2d[
    #             Orient.VERTICAL
    #         ] = self.__gtvt_selected_slices_3d[SAGITTAL]
    #         self.__total_slices_count_2d[Orient.HORIZONTAL] = self.img_3d[
    #             CT
    #         ].shape[1]
    #         self.__total_slices_count_2d[Orient.VERTICAL] = self.img_3d[CT].shape[
    #             2
    #         ]

    #     elif self._plane == CORONAL:
    #         self.__gtvt_selected_slices_2d[
    #             Orient.HORIZONTAL
    #         ] = self.__gtvt_selected_slices_3d[TRANSVERSE]
    #         self.__total_slices_count_2d[Orient.HORIZONTAL] = self.img_3d[
    #             CT
    #         ].shape[0]
    #         self.__gtvt_selected_slices_2d[
    #             Orient.VERTICAL
    #         ] = self.__gtvt_selected_slices_3d[SAGITTAL]
    #         self.__total_slices_count_2d[Orient.VERTICAL] = self.img_3d[CT].shape[
    #             2
    #         ]

    #     elif self._plane == SAGITTAL:
    #         self.__gtvt_selected_slices_2d[
    #             Orient.HORIZONTAL
    #         ] = self.__gtvt_selected_slices_3d[TRANSVERSE]
    #         self.__total_slices_count_2d[Orient.HORIZONTAL] = self.img_3d[
    #             CT
    #         ].shape[0]
    #         self.__gtvt_selected_slices_2d[
    #             Orient.VERTICAL
    #         ] = self.__gtvt_selected_slices_3d[CORONAL]
    #         self.__total_slices_count_2d[Orient.VERTICAL] = self.img_3d[CT].shape[
    #             1
    #         ]

    def _reset_cur_slice_id(self):
        if self._gtvs_center is not None:
            self.cur_slice_id[TRANSVERSE] = self._gtvs_center[0]
            self.cur_slice_id[CORONAL] = self._gtvs_center[1]
            self.cur_slice_id[SAGITTAL] = self._gtvs_center[2]

    def __switch_color_enhance_slider_bars(self):
        # hide and show sliders
        if self.display_mode() == PLANE_FIXED:
            show_list = [TRANSVERSE, CORONAL, SAGITTAL]
            hide_list = [CT, PT, MR1, MR2]
        else:
            show_list = [CT, PT, MR1, MR2]
            hide_list = [TRANSVERSE, CORONAL, SAGITTAL]

        for i in show_list:
            if self._radio_btn[COLOR_ENHANCE][i].isChecked():
                self._slider[BRIGHT][i].show()
                self._slider[CONTRAST][i].show()
            else:
                self._slider[BRIGHT][i].hide()
                self._slider[CONTRAST][i].hide()
        for i in hide_list:
            self._slider[BRIGHT][i].hide()
            self._slider[CONTRAST][i].hide()

        # update text label
        if self.display_mode() == PLANE_FIXED:
            for i in [TRANSVERSE, CORONAL, SAGITTAL]:
                if self._radio_btn[COLOR_ENHANCE][i].isChecked():
                    key_word = i.capitalize()
        else:
            for i in [CT, PT, MR1, MR2]:
                if self._radio_btn[COLOR_ENHANCE][i].isChecked():
                    if i == CT:
                        key_word = "CT"
                    elif i == PT:
                        key_word = "PT"
                    elif i == MR1:
                        key_word = "MR-T1"
                    elif i == MR2:
                        key_word = "MR-T2"

        self._text_label[BRIGHT].setText(
            "{} ({})".format(BRIGHT.capitalize(), key_word)
        )
        self._text_label[CONTRAST].setText(
            "{} ({})".format(CONTRAST.capitalize(), key_word)
        )

    def _clear_img_qlabels(self):
        for i in [CT, PT, MR1, MR2]:
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
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, BASELINE)
        fold_dir = Dir.get_sub_dirs(baseline_dir, key_word="fold=", full_path=True)[0]
        baseline_dataset_ver = Json.load(os.path.join(fold_dir, "hyper.json"))[
            DATASET_VER
        ]

        # set dataset dir based on current patient
        if self._cur_patient in self._patients["au.test.inter"]:
            self._dataset_ver = baseline_dataset_ver
            self._dataset_part = DatasetPart.TEST_INTER

        elif self._cur_patient in self._patients["au.test.exter"]:
            self._dataset_ver = baseline_dataset_ver
            self._dataset_part = DatasetPart.TEST_EXTER

        elif self._cur_patient in self._patients["mda.test"]:
            self._dataset_ver = MDA
            self._dataset_part = DatasetPart.TEST
        else:
            Debug.error_exit("cant find current patient in test patients")

        # set dataset dir and nii spacing
        self._dataset_dir = g.DATASET_DIR[self._dataset_ver]
        self._nii_spacing = g.NII_SPACING[self._dataset_ver]

    def _fill_combox_patient(self):
        combox_patients = Dir.get_sub_dirs(
            os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, "baseline", "patients")
        )
        # from "patient=123" to "123"
        for i in range(len(combox_patients)):
            combox_patients[i] = combox_patients[i][len("patient=") :]

        combox_patients.find_identical_items(self._patients.to_list())
        combox_patients.sort()
        self._combox[PATIENT].addItems(combox_patients)
        self._combox[PATIENT].setEnabled(True)
        return combox_patients

    def _load_baseline_data(self):
        # self._reset_zoomin()
        self.__clear_scores()
        self._clear_img_3d()
        self._clear_img_qlabels()

        # run this after current text of baseline combox is confirmed
        self._enable_arrow_btns(BASELINE)

        self._baseline_id = self._combox[BASELINE].currentText()

        # reset comboboxes
        for i in [PATIENT, IDL_GTVT, IDL_GTVN]:
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
        self._load_patient_data(idx=None, reset_patient=reset_patient)

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
        img_path = Dict()
        img_path[CT] = "CT"
        img_path[PT] = "PT"
        img_path[MR1] = "T1dr"
        img_path[MR2] = "T2dr"
        for i in [CT, PT, MR1, MR2]:
            img_path[i] = "HNCDL_{}_{}.nii".format(self._cur_patient, img_path[i])
            img_path[i] = os.path.join(self._dataset_dir, img_path[i])
            self.img_3d[i] = self._load_3d_img(img_path[i])

    def _load_patient_data(self, idx: int = None, reset_patient: bool = True):
        # triggered by:
        # (1) patient combox update
        # (2) baseline combox update, but can not find cur patient in new baseline dir
        if reset_patient is True:
            self._cur_patient = self._combox[PATIENT].currentText()
            # self._reset_zoomin()

        # triggered by baseline combox update, and find cur patient in new baseline dir
        else:
            self._combox[PATIENT].setCurrentText(self._cur_patient)

        # run these after patient combox current text is set up
        self._enable_arrow_btns(PATIENT)
        self._load_dataset_dir_and_nii_spacing()

        # reset comboboxes
        for i in [IDL_GTVT, IDL_GTVN]:
            self._combox[i].clear()
            self._combox[i].setEnabled(False)
            self._arrow_btn["prev.{}".format(i)].setEnabled(False)
            self._arrow_btn["next.{}".format(i)].setEnabled(False)

        # fill idl comboboxes
        for i in [IDL_GTVT, IDL_GTVN]:
            combox_items = [BASELINE]
            # get all round folder under current patient folder
            for idl_result_dir in Dir.get_sub_dirs(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id),
                key_word=i,
                full_path=True,
            ):
                if Path(idl_result_dir).name == "idl.gtvn_real.idl":
                    continue
                patient_dir = os.path.join(
                    idl_result_dir,
                    "patients",
                    "patient={}".format(self._cur_patient),
                )
                if os.path.exists(patient_dir):
                    round_folders = Dir.get_sub_dirs(
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
        self._reset_cur_slice_id()

        # choose idl automatically
        # try not to reset idl id/round when patient is changed
        for gtv in [GTVT, GTVN]:
            if self._idl_id[gtv] == BASELINE:
                reset_id = False
            elif (
                os.path.join(self._idl_id[gtv], self._idl_round[gtv])
                not in self._combox["idl.{}".format(gtv)].currentText()
            ):
                reset_id = True
            else:
                reset_id = False
            # refresh imgs after idl.gtvn is chosen
            self._load_idl_gtv_data(gtv=gtv, reset_id=reset_id, refresh_imgs=False)

        self.refresh_img_qlabels()

    # load labels and gtvs gravity center
    def __load_labels(self):
        labels = Img.load_labels(
            dataset_dir=self._dataset_dir,
            patient=self._cur_patient,
            nii_load_func=self._load_3d_img,
        )
        # load gtvt and gtvn
        for gtv in [GTVT, GTVN]:
            self.img_3d["{}.label".format(gtv)] = labels[gtv]
        # load gtvs gravity center: (d,h,w)
        self._gtvs_center = list(measurements.center_of_mass(labels[GTVS]))
        # float to int
        for i in range(len(self._gtvs_center)):
            self._gtvs_center[i] = round(self._gtvs_center[i])

    def _load_idl_gtvt_data(
        self, idx: int = None, reset_id: bool = True, refresh_imgs=True
    ):
        self._load_idl_gtv_data(gtv=GTVT, reset_id=reset_id, refresh_imgs=refresh_imgs)

    def _load_idl_gtvn_data(
        self, idx: int = None, reset_id: bool = True, refresh_imgs=True
    ):
        self._load_idl_gtv_data(gtv=GTVN, reset_id=reset_id, refresh_imgs=refresh_imgs)

    def __clear_gtvt_selected_slices_3d(self):
        self.__gtvt_selected_slices_3d = Dict()
        for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
            self.__gtvt_selected_slices_3d[plane] = List()

    # _load_idl_gtvt_data and _load_idl_gtvn_data will share this function
    def _load_idl_gtv_data(
        self, gtv: str, reset_id: bool = True, refresh_imgs: bool = True
    ):
        # triggered by:
        # (1) idl combox update
        # (2) patient combox update, but can not find cur patient in idl dir
        if reset_id is True:
            combox_item = self._combox["idl.{}".format(gtv)].currentText()
            if combox_item == BASELINE:
                self._idl_id[gtv] = BASELINE
                self._idl_round[gtv] = "round=00"
            else:
                self._idl_id[gtv] = combox_item[: combox_item.index("/")]
                self._idl_round[gtv] = combox_item[combox_item.index("/") + 1 :]
            # self._reset_zoomin()

        # triggered by patient combox update, and find cur patient in idl.gtvn dir
        else:
            if self._idl_id[gtv] == BASELINE:
                self._combox["idl.{}".format(gtv)].setCurrentText(BASELINE)
            else:
                self._combox["idl.{}".format(gtv)].setCurrentText(
                    os.path.join(self._idl_id[gtv], self._idl_round[gtv])
                )

        # run this after idl gtvn combox is filled
        self._enable_arrow_btns("idl.{}".format(gtv))

        # load data (pred/clicks/selected_slices)
        # baseline
        if self._idl_id[gtv] == BASELINE:
            pred_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                BASELINE,
                "patients",
                "patient={}".format(self._cur_patient),
                "{}_pred.nii.gz".format(gtv),
            )
            # clear idl.gtvt data
            if gtv == GTVT:
                for i in [CLICK, ANNOTATION, CORRECTION]:
                    self.img_3d["gtvt.{}".format(i)] = None
                self.__clear_gtvt_selected_slices_3d()
                # self.__refresh_gtvt_selected_slices_2d()
            # clear idl.gtvn data
            elif gtv == GTVN:
                for i in [CLICKS, CORRECTION]:
                    self.img_3d["gtvn.{}".format(i)] = None

        # idl.gtvt/gtvn
        else:
            cur_patient_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id[gtv],
                "patients",
                "patient={}".format(self._cur_patient),
            )
            cur_round_dir = os.path.join(
                cur_patient_dir,
                self._idl_round[gtv],
            )
            pred_path = os.path.join(cur_round_dir, "{}_pred.nii.gz".format(gtv))

            # load gtvt data
            if gtv == GTVT:
                # load gtvt nii
                for i in [CLICK, ANNOTATION, CORRECTION]:
                    nii_path = os.path.join(cur_round_dir, "gtvt_{}.nii.gz".format(i))
                    if os.path.exists(nii_path):
                        self.img_3d["gtvt.{}".format(i)] = self._load_3d_img(
                            nii_path, binary=True
                        )
                    else:
                        self.img_3d["gtvt.{}".format(i)] = None
                # load gtvt selected slices (3d)
                selected_slices_json_path = os.path.join(
                    cur_patient_dir,
                    "selected_slices.json",
                )
                if (
                    not os.path.exists(selected_slices_json_path)
                    or self._idl_round[GTVT] == "round=00"
                ):
                    self.__clear_gtvt_selected_slices_3d()
                else:
                    self.__gtvt_selected_slices_3d = Json.load(
                        selected_slices_json_path
                    )
                    for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
                        selected_slices_list = List()
                        for round_num in self.__gtvt_selected_slices_3d[plane]:
                            selected_slices_list += List(
                                self.__gtvt_selected_slices_3d[plane][round_num]
                            )
                            if (round_num) == self._idl_round[GTVT]:
                                break
                        # str to int
                        for i in range(len(selected_slices_list)):
                            selected_slices_list[i] = int(selected_slices_list[i])
                        self.__gtvt_selected_slices_3d[plane] = selected_slices_list

                # refresh gtvt selected slices (2d)
                # after gtvt selected slices (3d) is loaded
                # self.__refresh_gtvt_selected_slices_2d()

            # load gtvn data
            elif gtv == GTVN:
                # load gtvn nii
                for i in [CLICKS, CORRECTION]:
                    nii_path = os.path.join(cur_round_dir, "gtvn_{}.nii.gz".format(i))
                    if os.path.exists(nii_path):
                        self.img_3d["gtvn.{}".format(i)] = self._load_3d_img(
                            nii_path, binary=True
                        )
                    else:
                        self.img_3d["gtvn.{}".format(i)] = None

        # load preds
        self.img_3d["{}.pred".format(gtv)] = self._load_3d_img(pred_path, binary=True)

        # load baseline scores
        if self._idl_id[gtv] == BASELINE:
            gtvn_score_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                BASELINE,
                "inference_{}_{}.json".format(self._dataset_ver, self._dataset_part),
            )
            if os.path.exists(gtvn_score_path):
                gtvn_score = Json.load(gtvn_score_path)
                for metric in [DSC, MSD, HD95]:
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
                for metric in [DSC, MSD, HD95]:
                    self.__scores[gtv][metric] = gtvn_score[
                        "patient={}".format(self._cur_patient)
                    ][metric][self._idl_round[gtv]]

        if refresh_imgs:
            self.refresh_img_qlabels()

    def _load_prev_baseline_data(self):
        idx = self._combox[BASELINE].currentIndex() - 1
        if idx < 0:
            return
        prev_baseline = self._combox[BASELINE].itemText(idx)
        self._combox[BASELINE].setCurrentText(prev_baseline)
        self._load_baseline_data()

    def _load_next_baseline_data(self):
        idx = self._combox[BASELINE].currentIndex() + 1
        if idx > self._combox[BASELINE].count() - 1:
            return
        next_baseline = self._combox[BASELINE].itemText(idx)
        self._combox[BASELINE].setCurrentText(next_baseline)
        self._load_baseline_data()

    def _load_prev_idl_gtvn_data(self):
        idx = self._combox[IDL_GTVN].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvn = self._combox[IDL_GTVN].itemText(idx)
        self._combox[IDL_GTVN].setCurrentText(prev_idl_gtvn)
        self._load_idl_gtvn_data()

    def _load_next_idl_gtvn_data(self):
        idx = self._combox[IDL_GTVN].currentIndex() + 1
        if idx > self._combox[IDL_GTVN].count() - 1:
            return
        next_idl_gtvn = self._combox[IDL_GTVN].itemText(idx)
        self._combox[IDL_GTVN].setCurrentText(next_idl_gtvn)
        self._load_idl_gtvn_data()

    def _load_prev_idl_gtvt_data(self):
        idx = self._combox[IDL_GTVT].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvt = self._combox[IDL_GTVT].itemText(idx)
        self._combox[IDL_GTVT].setCurrentText(prev_idl_gtvt)
        self._load_idl_gtvt_data()

    def _load_next_idl_gtvt_data(self):
        idx = self._combox[IDL_GTVT].currentIndex() + 1
        if idx > self._combox[IDL_GTVT].count() - 1:
            return
        next_idl_gtvt = self._combox[IDL_GTVT].itemText(idx)
        self._combox[IDL_GTVT].setCurrentText(next_idl_gtvt)
        self._load_idl_gtvt_data()

    def _load_prev_patient_data(self):
        idx = self._combox[PATIENT].currentIndex() - 1
        if idx < 0:
            return
        prev_patient = self._combox[PATIENT].itemText(idx)
        self._combox[PATIENT].setCurrentText(prev_patient)
        self._load_patient_data()

    def _load_next_patient_data(self):
        idx = self._combox[PATIENT].currentIndex() + 1
        if idx > self._combox[PATIENT].count() - 1:
            return
        next_patient = self._combox[PATIENT].itemText(idx)
        self._combox[PATIENT].setCurrentText(next_patient)
        self._load_patient_data()

    def __gray_to_rgb(self, gray_img: ndarray):
        # rgb_img = np.uint8((gray_img - gray_img.min()) / gray_img.ptp() * 255.0)
        gray_img = cv2.convertScaleAbs(gray_img, alpha=255.0)
        # after cv2.cvtColor, rgb_img has 3 channels, but is still numpy
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        return rgb_img

    def __gray_to_colormap(self, gray_img: ndarray):
        gray_img = cv2.convertScaleAbs(1 - gray_img, alpha=255.0)
        color_map = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        return color_map

    def __refresh_img_qlabel_trasverse(self):
        self.refresh_img_qlabels(img_name=TRANSVERSE)

    # replay_mode=True will show all contours
    # otherwise correction and annotation will cover pred
    def refresh_img_qlabels(
        self,
        slider_value=None,  # get this value when function is triggerrd by QSlider, ignore it
        replay_mode: bool = True,
        img_name=None,
    ):
        if self.img_3d[CT] is None:
            return

        if img_name is not None:
            img_name_list = [img_name]
        else:
            if self.display_mode() == PLANE_FIXED:
                img_name_list = [TRANSVERSE, CORONAL, SAGITTAL]
            else:
                img_name_list = [CT, PT, MR1, MR2]

        # load rgb imgs
        for img_name in img_name_list:
            modal = self.img_qlabel[img_name].modal

            # plane fixed mode
            if self.display_mode() == PLANE_FIXED:
                cur_slice_id = self.cur_slice_id[self.img_qlabel[img_name].plane]
                if img_name == TRANSVERSE:
                    ct_slice = self.img_3d[CT][cur_slice_id, :, :]
                    pt_slice = self.img_3d[PT][cur_slice_id, :, :]
                elif img_name == CORONAL:
                    slice_2d = self.img_3d[modal][:, cur_slice_id, :]
                elif img_name == SAGITTAL:
                    slice_2d = self.img_3d[modal][:, :, cur_slice_id]

            # modality fixed mode
            else:
                cur_slice_id = self.cur_slice_id[self.img_qlabel[img_name].plane]
                if self.img_qlabel[img_name].plane == TRANSVERSE:
                    slice_2d = self.img_3d[img_name][cur_slice_id, :, :]
                elif self.img_qlabel[img_name].plane == CORONAL:
                    slice_2d = self.img_3d[img_name][:, cur_slice_id, :]
                elif self.img_qlabel[img_name].plane == SAGITTAL:
                    slice_2d = self.img_3d[img_name][:, :, cur_slice_id]

            if img_name == TRANSVERSE:
                pt_slice = self.__gray_to_colormap(pt_slice)
                ct_slice = self.__gray_to_rgb(ct_slice)
                alpha = self._slider[CT_PT_MIX].value() / 100
                rgb_img = cv2.addWeighted(
                    src1=pt_slice,
                    alpha=alpha,
                    src2=ct_slice,
                    beta=1 - alpha,
                    gamma=0,
                )
            else:
                if modal == PT:
                    rgb_img = self.__gray_to_colormap(slice_2d)
                else:
                    rgb_img = self.__gray_to_rgb(slice_2d)

            # brightness and contrast
            # cv2.addWeighted: dst = src1 * alpha + src2 * beta + gamma
            rgb_img = cv2.addWeighted(
                src1=rgb_img,
                alpha=self._slider[CONTRAST][img_name].value() / 100,
                src2=np.zeros_like(rgb_img),
                beta=0,
                gamma=self._slider[BRIGHT][img_name].value(),
            )

            # # add mask to gtvt selected slices
            # rgb_img_zeros = np.zeros((rgb_img.shape), dtype=np.uint8)
            # selected_slices_mask = None
            # for orient in [Orient.HORIZONTAL, Orient.VERTICAL]:
            #     for gtvt_selected_slice_2d in self.__gtvt_selected_slices_2d[orient]:
            #         # all images are reversed in transverse plane
            #         if self._plane != TRANSVERSE and orient == Orient.HORIZONTAL:
            #             slice_pos = (
            #                 self.__total_slices_count_2d[orient]
            #                 - gtvt_selected_slice_2d
            #             )
            #         # 1mm images are reversed in sagittal plane
            #         elif (
            #             self._plane != SAGITTAL
            #             and orient == Orient.VERTICAL
            #             and (
            #                 self._dataset_ver == AU_1MM
            #                 or self._dataset_ver == MDA
            #             )
            #         ):
            #             slice_pos = (
            #                 self.__total_slices_count_2d[orient]
            #                 - gtvt_selected_slice_2d
            #             )
            #         else:
            #             slice_pos = gtvt_selected_slice_2d

            #         if orient == Orient.HORIZONTAL:
            #             x1 = 0
            #             y1 = slice_pos
            #             x2 = rgb_img.shape[1] - 1
            #             y2 = slice_pos
            #         elif orient == Orient.VERTICAL:
            #             x1 = slice_pos
            #             y1 = 0
            #             x2 = slice_pos
            #             y2 = rgb_img.shape[0] - 1

            #         cur_slice_mask = cv2.rectangle(
            #             img=rgb_img_zeros,
            #             pt1=(x1, y1),
            #             pt2=(x2, y2),
            #             color=self._color[GTVT_ANNOTATION],
            #             thickness=-1,
            #         )
            #         if selected_slices_mask is None:
            #             selected_slices_mask = cur_slice_mask
            #         else:
            #             selected_slices_mask += cur_slice_mask

            # if selected_slices_mask is not None:
            #     rgb_img = cv2.addWeighted(
            #         src1=rgb_img,
            #         alpha=1,
            #         src2=selected_slices_mask,
            #         beta=1,  # 0.5,
            #         gamma=0,
            #     )

            # resize and fit img qlabel
            rgb_img, _ = self._fit_img_qlabel(rgb_img, self.img_qlabel[img_name])
            if img_name == CT:
                self._rgb_img_roi = _

            # blur after _fit_img_qlabel will gain better effect
            rgb_img = cv2.GaussianBlur(rgb_img, (3, 3), cv2.BORDER_DEFAULT)

            # replay mode, place the name of img on the top layer at the end of the list
            if replay_mode:
                seg_name_list = [
                    GTVN_LABEL,
                    GTVT_LABEL,
                    GTVN_PRED,
                    GTVT_PRED,
                    GTVT_ANNOTATION,
                    GTVN_CORRECTION,
                    GTVT_CORRECTION,
                    GTVN_CLICKS,
                    GTVT_CLICK,
                ]
            # idl mode, correction > annotation > pred
            else:
                seg_name_list = [
                    GTVN_PRED_FINAL,
                    GTVT_PRED_FINAL,
                    GTVN_CLICKS,
                    GTVT_CLICK,
                ]

            # draw label and pred contour
            for seg_name in seg_name_list:
                if self.img_3d[seg_name] is None:
                    continue

                # load data of current slice
                if self.img_qlabel[img_name].plane == SAGITTAL:
                    segment = self.img_3d[seg_name][:, :, self.cur_slice_id[SAGITTAL]]
                elif self.img_qlabel[img_name].plane == CORONAL:
                    segment = self.img_3d[seg_name][:, self.cur_slice_id[CORONAL], :]
                elif self.img_qlabel[img_name].plane == TRANSVERSE:
                    segment = self.img_3d[seg_name][self.cur_slice_id[TRANSVERSE], :, :]

                segment = segment.astype(np.uint8)

                # skip if current contour img is empty
                if seg_name in [
                    GTVN_CORRECTION,
                    GTVT_CORRECTION,
                    GTVT_ANNOTATION,
                ]:
                    # perfomr erosion to remove overlap of 3 different planes
                    kernel = np.ones((3, 3), np.uint8)
                    eroded_segment = cv2.erode(segment, kernel, iterations=1)
                    if eroded_segment.max() <= 0:
                        continue
                else:
                    if segment.max() <= 0:
                        continue

                segment, _ = self._fit_img_qlabel(segment, self.img_qlabel[img_name])

                # points, higher thickness (otherwise cant see the points)
                if seg_name == GTVT_CLICK or seg_name == GTVN_CLICKS:
                    thickness = 7
                # contours, lower thickness
                else:
                    thickness = 2
                    # blur, make the contours looks better on the UI
                    # blur after _fit_img_qlabel()
                    segment = cv2.GaussianBlur(segment, (7, 7), cv2.BORDER_DEFAULT)

                # find and draw contours
                contours, _ = cv2.findContours(
                    segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                rgb_img = cv2.drawContours(
                    image=rgb_img,
                    contours=contours,
                    contourIdx=-1,
                    color=self._color[seg_name],
                    thickness=thickness,
                )

            rgb_img_height = rgb_img.shape[0]
            rgb_img_width = rgb_img.shape[1]
            rgb_img_chan = rgb_img.shape[2]

            # ndarray to qimage
            qimg = QImage(
                rgb_img,
                rgb_img_width,
                rgb_img_height,
                rgb_img_width * rgb_img_chan,
                QImage.Format_RGB888,
            )

            # top left text
            if img_name == TRANSVERSE or img_name == CT:
                self._add_score_on_qimg(qimg)
                self._add_msg_on_qimg(qimg)

            # bottom left text
            if img_name == TRANSVERSE or img_name == MR1:
                self._add_contour_description_on_qimg(qimg)

            self.img_qlabel[img_name].set_background(qimg)
            self.img_qlabel[img_name].update()

    def _add_contour_description_on_qimg(self, qimg: QImage):
        pos_x = [10, 65, 110]
        pos_y = [qimg.height() - 57, qimg.height() - 35, qimg.height() - 13]

        # label
        self._qimg_draw_text(
            qimg=qimg,
            text="Label:",
            pos=(pos_x[0], pos_y[0]),
            color=self._color["green"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVt",
            pos=(pos_x[1], pos_y[0]),
            color=self._color[GTVT_LABEL],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVn",
            pos=(pos_x[2], pos_y[0]),
            color=self._color[GTVN_LABEL],
        )

        # pred
        self._qimg_draw_text(
            qimg=qimg,
            text="Pred:",
            pos=(pos_x[0], pos_y[1]),
            color=self._color["green"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVt",
            pos=(pos_x[1], pos_y[1]),
            color=self._color[GTVT_PRED],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVn",
            pos=(pos_x[2], pos_y[1]),
            color=self._color[GTVN_PRED],
        )

        # user input
        self._qimg_draw_text(
            qimg=qimg,
            text="User:",
            pos=(pos_x[0], pos_y[2]),
            color=self._color["green"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="Init",
            pos=(pos_x[1], pos_y[2]),
            color=self._color[GTVT_ANNOTATION],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="Correction",
            pos=(pos_x[2], pos_y[2]),
            color=self._color[GTVT_CORRECTION],
        )

    def _add_msg_on_qimg(self, qimg: QImage):
        pass

    def _add_score_on_qimg(self, qimg: QImage):
        pos_y = 25

        for metric in [DSC, MSD, HD95]:
            pos_x = 10

            # "DSC/MSD/HD95: "
            text = metric.upper() + ": "
            self._qimg_draw_text(
                qimg=qimg,
                text=text,
                pos=(pos_x, pos_y),
                color=self._color["green"],
            )
            # load scores
            for i in [GTVT, GTVN]:
                # text
                if Value.is_number(self.__scores[i][metric]):
                    if metric == DSC:
                        text = "{:.2f}".format(self.__scores[i][metric])
                    else:
                        text = "{:.1f}".format(self.__scores[i][metric])
                else:
                    text = "NaN"
                # mod x pos
                if i == GTVT:
                    pos_x += 55
                else:
                    pos_x += 50
                # draw text
                self._qimg_draw_text(
                    qimg=qimg,
                    text=text,
                    pos=(pos_x, pos_y),
                    color=self._color["{}.pred".format(i)],
                )
            # mod y pos
            pos_y += 20

    def get_cur_patient_idl_step(self):
        return None

    def _qimg_draw_text(
        self,
        qimg,
        text: str,
        pos: tuple,
        color: tuple,
        line_gap: int = 20,
    ):
        font = QFont("Arial", 12)
        font.setBold(True)
        painter = QPainter(qimg)
        painter.setFont(font)
        r, g, b = color
        alpha = 255

        x = pos[0]
        for i, line in enumerate(text.split("\n")):
            y = pos[1] + i * line_gap

            # draw outline
            # outline_color = QColor("black")
            painter.setPen(Qt.black)
            # Adjust for desired thickness
            offsets = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
            for x_off, y_off in offsets:
                painter.drawText(x + x_off, y + y_off, line)

            # draw text
            painter.setPen(QColor(r, g, b, alpha))
            painter.drawText(x, y, line)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__resize_img_qlabels()
        self._refresh_side_bar()
        self.refresh_img_qlabels()

    def __resize_img_qlabels(self):
        gap = 1
        # pos: w0 w1 h0 h1
        pos = Dict()
        pos["w"] = self.geometry().width() - self.__side_bar_width
        pos["h"] = self.geometry().height()
        for i in ["w", "h"]:
            double_size = pos[i] - gap * 3
            pos.pop(i)
            pos[i][0] = double_size // 2
            pos[i][1] = double_size // 2
            if double_size % 2 != 0:
                pos[i][0] += 1

        # pos: x0 x1 y0 y1
        for i in ["x", "y"]:
            pos[i][0] = gap
            if i == "x":
                j = "w"
            else:
                j = "h"
            pos[i][1] = pos[j][0] + gap * 2

        # ct
        self.img_qlabel[CT].setGeometry(
            QRect(pos["x"][0], pos["y"][0], pos["w"][0], pos["h"][0])
        )
        # transverse
        self.img_qlabel[TRANSVERSE].setGeometry(
            QRect(pos["x"][0], pos["y"][0], pos["w"][0], pos["h"][0] + 1 + pos["h"][1])
        )
        # pt / coronal
        for i in [PT, CORONAL]:
            self.img_qlabel[i].setGeometry(
                QRect(pos["x"][1], pos["y"][0], pos["w"][1], pos["h"][0])
            )
        # mr1
        self.img_qlabel[MR1].setGeometry(
            QRect(pos["x"][0], pos["y"][1], pos["w"][0], pos["h"][1])
        )
        # mr2 sagittal
        for i in [MR2, SAGITTAL]:
            self.img_qlabel[i].setGeometry(
                QRect(pos["x"][1], pos["y"][1], pos["w"][1], pos["h"][1])
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
