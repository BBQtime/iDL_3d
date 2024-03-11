import os
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
from scipy.ndimage import measurements
from str_lib import DatasetPart, DatasetVer, DisplayMode, Metric, Modal, Plane
from superqt import QCollapsible
from ui_custom_combox import CustomComboBox
from ui_img_frame import ImgFrame
from ui_toggle_btn import ToggleButton


class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # load setting
        ui_setting = Json.load(os.path.join(g.PROJ_DIR, "settings_ui.json"))
        self._init_data(ui_setting)
        self._init_color(ui_setting)  # before init_widgets()
        self._init_widgets(ui_setting)  # after _init_data()
        # self.__init_zoomin()
        self._load_baseline_data()  # load first baseline result

    def _init_patients(self):
        # load test set patients of all datasets
        self._patients = Dict()
        for i in [DatasetVer.AU, DatasetVer.OBS_STUDY, DatasetVer.MDA]:
            dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH[i])
            self._patients[i] = List(dataset_split[DatasetPart.TEST])

    def _init_data(
        self,
        ui_setting: Dict,  # this is for idl_window
    ):
        self._debug_mode = ui_setting["debug.mode"]

        # =1 for replay window
        self.interpolation_step = 1

        self._init_patients()

        # init baseline id and cur patient
        self._baseline_id = None
        self._cur_patient = None

        # init cur_slice_id dict and gtvs center
        self.cur_slice_id = Dict()
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.cur_slice_id[i] = 0  # starts from 0
        self._gtvs_center = None

        # show contour or not, switch this by key "X"
        self.__show_contour = True

        # init idl_id and idl_round
        self._idl_id = Dict()
        self._idl_round = Dict()
        for i in ["gtvt", "gtvn"]:
            self._idl_id[i] = "baseline"
            self._idl_round[i] = "round=00"

        # dataset and nii spacing
        self.dataset_ver = None
        self._dataset_dir = None  # au / obs.study / mda

        # init score_dict
        self.__scores = Dict()
        self.__clear_scores()

        # img_3d save the ndarray of imgs/preds/delineation/corrections
        self.img_3d = Dict()
        self._clear_img_3d()

        # save the original/zoomed 2d rgb imgs (with contours)
        self._origin_rgb = Dict()
        self._zoomed_rgb = Dict()
        self._contoured_rgb = Dict()
        for i in [
            Plane.TRANSVERSE,
            Plane.CORONAL,
            Plane.SAGITTAL,
            Modal.CT,
            Modal.PT,
            Modal.MR1,
            Modal.MR2,
        ]:
            self._origin_rgb[i] = None
            self._zoomed_rgb[i] = None
            self._contoured_rgb[i] = None

        self.__clear_gtvt_selected_slices_3d()

    def _add_border(self, input_widget: QtWidgets.QWidget):
        random_name = Value.random_str()
        input_widget.setObjectName(random_name)
        input_widget.setStyleSheet(f"#{random_name} {{border: 2px solid gray;}}")

    def _clear_img_3d(self):
        for i in [
            Modal.CT,
            Modal.PT,
            Modal.MR1,
            Modal.MR2,
            "gtvt.label",
            "gtvn.label",
            "gtvt.pred",
            "gtvn.pred",
            "gtvt.click",
            "gtvn.clicks",
            "gtvt.delineation",
            "gtvt.correction",
            "gtvn.correction",
            "gtvt.pred.final",
            "gtvn.pred.final",
        ]:
            self.img_3d[i] = None

    def __clear_scores(self):
        for i in ["gtvt", "gtvn"]:
            self.__scores[i][Metric.DSC] = None
            self.__scores[i][Metric.MSD] = None
            self.__scores[i][Metric.HD95] = None

    def _init_color(self, ui_setting: Dict):
        self.color = Dict()
        for i in [
            "black",
            "red",
            "green",
            "magenta",
            "cyan",
            "blue",
            "yellow",
            "orange",
        ]:
            self.color[i] = List(ui_setting["color.def"][i])
            self.color[i] = tuple(int(k) for k in self.color[i])

        for i in [
            "gtvt.pred",
            "gtvt.label",
            "gtvn.pred",
            "gtvn.label",
            "gtvt.correction",
            "gtvn.correction",
        ]:
            self.color[i] = self.color[ui_setting["color.contour"][i]]

        # colors for replay mode only
        for i in ["gtvt.click", "gtvn.clicks", "gtvt.delineation"]:
            self.color[i] = self.color[
                ui_setting["color.contour"]["{}.replay".format(i)]
            ]

    def setupUi(self, Core):
        Core.setObjectName("Core")
        self._central_widget = QtWidgets.QWidget(Core)
        # self._central_widget.setObjectName("_central_widget")
        Core.setCentralWidget(self._central_widget)
        self.retranslateUi(Core)
        QtCore.QMetaObject.connectSlotsByName(Core)

    def retranslateUi(self, Core):
        _translate = QtCore.QCoreApplication.translate
        Core.setWindowTitle(_translate("Core", "Interactive Deep-learning Tool"))

    def _init_widgets_img_frames(self):
        self.img_frame = Dict()
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, Qt.black)

        for i in [
            Modal.CT,
            Modal.PT,
            Modal.MR1,
            Modal.MR2,
            Plane.TRANSVERSE,
            Plane.CORONAL,
            Plane.SAGITTAL,
        ]:
            self.img_frame[i] = ImgFrame(self._central_widget)
            self.img_frame[i].setObjectName("")
            # fill background with black
            self.img_frame[i].setPalette(pal)
            self.img_frame[i].setAutoFillBackground(True)

        # fixed plane
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.img_frame[i].plane = i

        # fixed modal
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_frame[i].modal = i

        self.img_frame[Plane.TRANSVERSE].modal = "mix"

    def _init_widgets_combox(self):
        combox_height = 30 if g.is_linux() else 45
        btn_height = 30 if g.is_linux() else 38

        self.combox = Dict()
        for i in ["baseline", "patient", "idl.gtvt", "idl.gtvn"]:
            self.combox[i] = CustomComboBox()
            self.combox[i].setFixedHeight(combox_height)
            if i in ["patient", "idl.gtvt", "idl.gtvn"]:
                self.combox[i].setEnabled(False)

        # fill combox baseline
        baseline_id_list = Dir.get_sub_dirs(
            g.TRAIN_RESULTS_DIR, key_word="baseline_", shuffle=False
        )
        self.combox["baseline"].addItems(baseline_id_list)

        # set observer study baseline id as default
        for baseline_id in baseline_id_list:
            if "obs.study" in baseline_id:
                obs_study_baseline_id = baseline_id
        self.combox["baseline"].setCurrentText(obs_study_baseline_id)

        # arrow buttons
        self._arrow_btn = Dict()
        for i in ["prev", "next"]:
            for j in ["baseline", "patient", "idl.gtvt", "idl.gtvn"]:
                self._arrow_btn["{}.{}".format(i, j)] = QtWidgets.QToolButton()
                self._arrow_btn["{}.{}".format(i, j)].setFixedWidth(btn_height)
                self._arrow_btn["{}.{}".format(i, j)].setFixedHeight(btn_height)

        # set arrow buttons initial state
        for i in ["baseline", "patient", "idl.gtvt", "idl.gtvn"]:
            self._arrow_btn["prev.{}".format(i)].setArrowType(Qt.LeftArrow)
            self._arrow_btn["next.{}".format(i)].setArrowType(Qt.RightArrow)
            if i in ["patient", "idl.gtvt", "idl.gtvn"]:
                self._arrow_btn["prev.{}".format(i)].setEnabled(False)
                self._arrow_btn["next.{}".format(i)].setEnabled(False)

        # collapse - baseline/patient/idl.gtvt/gtvn
        self._collap["baseline"] = QCollapsible("SELECT BASELINE")
        self._collap["patient"] = QCollapsible("SELECT PATIENT")
        self._collap["idl.gtvt"] = QCollapsible("SELECT IDL GTVT")
        self._collap["idl.gtvn"] = QCollapsible("SELECT IDL GTVN")
        for i in ["baseline", "patient", "idl.gtvt", "idl.gtvn"]:
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.setSpacing(1)
            h_layout.addWidget(self._arrow_btn["prev.{}".format(i)])
            h_layout.addWidget(self.combox[i])
            h_layout.addWidget(self._arrow_btn["next.{}".format(i)])
            container = QtWidgets.QWidget()
            container.setLayout(h_layout)
            self._add_border(container)
            self._collap[i].addWidget(container)
            self._collap[i].expand()

        # connect ctrls to functions
        self.combox["baseline"].activated.connect(self._load_baseline_data)
        self._arrow_btn["prev.baseline"].clicked.connect(self._load_prev_baseline_data)
        self._arrow_btn["next.baseline"].clicked.connect(self._load_next_baseline_data)
        self.combox["patient"].activated.connect(self.__on_combox_patient_clicked)
        self._arrow_btn["prev.patient"].clicked.connect(self._load_prev_patient_data)
        self._arrow_btn["next.patient"].clicked.connect(self._load_next_patient_data)
        self.combox["idl.gtvt"].activated.connect(self.__on_combox_idl_gtvt_clicked)
        self._arrow_btn["prev.idl.gtvt"].clicked.connect(self._load_prev_idl_gtvt_data)
        self._arrow_btn["next.idl.gtvt"].clicked.connect(self._load_next_idl_gtvt_data)
        self.combox["idl.gtvn"].activated.connect(self.__on_combox_idl_gtvn_clicked)
        self._arrow_btn["prev.idl.gtvn"].clicked.connect(self._load_prev_idl_gtvn_data)
        self._arrow_btn["next.idl.gtvn"].clicked.connect(self._load_next_idl_gtvn_data)

    def _init_widgets_color_enhance(self):
        # (1) radio buttons: ct/pt/mr1/mr2
        self._radio_group["color.enhance"] = QtWidgets.QButtonGroup()
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self._radio_btn["color.enhance"][i] = QtWidgets.QRadioButton()
            self._radio_btn["color.enhance"][i].setFixedHeight(g.TEXT_HEIGHT)
            self._radio_group["color.enhance"].addButton(
                self._radio_btn["color.enhance"][i]
            )
        # set text
        self._radio_btn["color.enhance"][Modal.CT].setText("CT")
        self._radio_btn["color.enhance"][Modal.PT].setText("PT")
        self._radio_btn["color.enhance"][Modal.MR1].setText("MR-T1")
        self._radio_btn["color.enhance"][Modal.MR2].setText("MR-T2")
        # set checked
        self._radio_btn["color.enhance"][Modal.CT].setChecked(True)

        # (2) text labels for slider bars
        for i in ["bright", "contrast"]:
            self._text_label[i] = QtWidgets.QLabel()
            self._text_label[i].setFixedHeight(g.TEXT_HEIGHT)
        self._text_label["bright"].setText("Brightness (CT)")
        self._text_label["contrast"].setText("Contrast (CT)")

        # (3) slider bars
        for i in ["bright", "contrast"]:
            for j in [
                Modal.CT,
                Modal.PT,
                Modal.MR1,
                Modal.MR2,
            ]:
                self._slider[i][j] = QtWidgets.QSlider()
                self._slider[i][j].setFixedHeight(g.SLIDER_HEIGHT)
                slider = self._slider[i][j]
                slider.setOrientation(Qt.Horizontal)
                if i == "bright":
                    slider.setMinimum(-128)
                    slider.setMaximum(128)
                    slider.setValue(0)
                elif i == "contrast":
                    slider.setMinimum(0)
                    slider.setMaximum(200)
                    slider.setValue(100)
                # only show ct slider bars
                if j != Modal.CT:
                    slider.hide()

        # v layout
        v_layout = QtWidgets.QVBoxLayout()
        # add text labels and slider bars
        for i in ["bright", "contrast"]:
            v_layout.addWidget(self._text_label[i])
            for j in [
                Modal.CT,
                Modal.PT,
                Modal.MR1,
                Modal.MR2,
            ]:
                v_layout.addWidget(self._slider[i][j])
        # add radio buttons: ct/pt/mr1/mr2
        h_layout = QtWidgets.QHBoxLayout()
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            h_layout.addWidget(self._radio_btn["color.enhance"][i])
        v_layout.addLayout(h_layout)

        # add final layout into collapsible space
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)
        # collapse
        self._collap["color.enhance"] = QCollapsible("COLOR ENHANCEMENT")
        self._collap["color.enhance"].addWidget(container)
        self._collap["color.enhance"].collapse()

        # connect widgets to functions
        for i in ["bright", "contrast"]:
            for j in [
                Modal.CT,
                Modal.PT,
                Modal.MR1,
                Modal.MR2,
            ]:
                self._slider[i][j].valueChanged.connect(
                    self.__color_enhance_slider_value_update
                )
        self._radio_group["color.enhance"].buttonClicked.connect(
            self.__switch_color_enhance_slider_bars
        )

    # this function is connected to widget, dont set input params to this function
    def __color_enhance_slider_value_update(self):
        # refresh origin_rgb as bright/contrast changed
        self.refresh_imgs()

    # this function is connected to widget, dont set input params to this function
    def _plane_fixed_mode_switch_modal(self):
        # update modality
        for modal in self._radio_btn[DisplayMode.PLANE_FIXED].keys():
            if self._radio_btn[DisplayMode.PLANE_FIXED][modal].isChecked():
                for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                    self.img_frame[plane].modal = modal
                break
        # update textlabel
        if self._radio_btn[DisplayMode.PLANE_FIXED][Modal.PT].isChecked():
            self._text_label["other.modal"].setText("PT")
        elif self._radio_btn[DisplayMode.PLANE_FIXED][Modal.MR1].isChecked():
            self._text_label["other.modal"].setText("MR-T1")
        elif self._radio_btn[DisplayMode.PLANE_FIXED][Modal.MR2].isChecked():
            self._text_label["other.modal"].setText("MR-T2")

        # refresh from origin_rgb as modality changed
        self.refresh_imgs()

    def display_mode(self):
        if self._toggle_btn.isChecked():
            return DisplayMode.PLANE_FIXED
        else:
            return DisplayMode.MODAL_FIXED

    def switch_display_mode(self):
        display_mode = self.display_mode()

        # img_frames: modalities
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            if display_mode == DisplayMode.PLANE_FIXED:
                self.img_frame[i].hide()
            else:
                self.img_frame[i].show()

        # img_frames: planes
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if display_mode == DisplayMode.PLANE_FIXED:
                self.img_frame[i].show()
            else:
                self.img_frame[i].hide()

        # plane fixed mode: text labels
        for i in [Modal.CT, "other.modal"]:
            if display_mode == DisplayMode.PLANE_FIXED:
                self._text_label[i].show()
            else:
                self._text_label[i].hide()

        # radio buttons: modal fixed mode
        for i in self._radio_btn[DisplayMode.MODAL_FIXED].keys():
            if display_mode == DisplayMode.PLANE_FIXED:
                self._radio_btn[DisplayMode.MODAL_FIXED][i].hide()
            else:
                self._radio_btn[DisplayMode.MODAL_FIXED][i].show()

        # ratio buttons: plane fixed mode
        for i in [Modal.PT, Modal.MR1, Modal.MR2]:
            if display_mode == DisplayMode.PLANE_FIXED:
                self._radio_btn[DisplayMode.PLANE_FIXED][i].show()
            else:
                self._radio_btn[DisplayMode.PLANE_FIXED][i].hide()

        # ct/pt mix slider
        if display_mode == DisplayMode.PLANE_FIXED:
            self._slider["mix"].show()
        else:
            self._slider["mix"].hide()

        self.reset_cur_slice_id()
        # refresh everything as brightness/contrast might changed
        self.refresh_imgs()
        self.refresh_crosses()

    # abstract function
    def reset_cur_slice_id(self):
        return

    # abstract function
    def refresh_crosses(self):
        return

    def _init_widgets_display_mode(self):
        # (1) toggle button
        self._toggle_btn = ToggleButton(is_checked=True)

        # (2) text: "Modality Fixed" and "Plane Fixed"
        for i in [DisplayMode.MODAL_FIXED, DisplayMode.PLANE_FIXED]:
            self._text_label[i] = QtWidgets.QLabel()
            self._text_label[i].setFixedHeight(self._toggle_btn.height())
        self._text_label[DisplayMode.MODAL_FIXED].setText("Modality Fixed")
        self._text_label[DisplayMode.MODAL_FIXED].setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        self._text_label[DisplayMode.PLANE_FIXED].setText("Plane Fixed")
        self._text_label[DisplayMode.PLANE_FIXED].setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )

        # (3) modality fixed mode: radio buttons
        self._radio_group[DisplayMode.MODAL_FIXED] = QtWidgets.QButtonGroup()
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self._radio_btn[DisplayMode.MODAL_FIXED][i] = QtWidgets.QRadioButton()
            self._radio_btn[DisplayMode.MODAL_FIXED][i].setFixedHeight(g.TEXT_HEIGHT)
            self._radio_btn[DisplayMode.MODAL_FIXED][i].setText(i.capitalize())
            self._radio_group[DisplayMode.MODAL_FIXED].addButton(
                self._radio_btn[DisplayMode.MODAL_FIXED][i]
            )
        # set checked
        self._radio_btn[DisplayMode.MODAL_FIXED][Plane.TRANSVERSE].setChecked(True)
        # connect ui to functions
        self._radio_group[DisplayMode.MODAL_FIXED].buttonClicked.connect(
            self.__on_modal_fixed_radio_group_clicked
        )

        # (4) plane fixed mode: radio buttons
        self._radio_group[DisplayMode.PLANE_FIXED] = QtWidgets.QButtonGroup()
        for i in [Modal.PT, Modal.MR1, Modal.MR2]:
            self._radio_btn[DisplayMode.PLANE_FIXED][i] = QtWidgets.QRadioButton()
            self._radio_btn[DisplayMode.PLANE_FIXED][i].setFixedHeight(g.TEXT_HEIGHT)
            self._radio_group[DisplayMode.PLANE_FIXED].addButton(
                self._radio_btn[DisplayMode.PLANE_FIXED][i]
            )
        # set checked
        self._radio_btn[DisplayMode.PLANE_FIXED][Modal.PT].setChecked(True)
        # set text
        self._radio_btn[DisplayMode.PLANE_FIXED][Modal.PT].setText("PT")
        self._radio_btn[DisplayMode.PLANE_FIXED][Modal.MR1].setText("MR-T1")
        self._radio_btn[DisplayMode.PLANE_FIXED][Modal.MR2].setText("MR-T2")
        # connect functions
        self._radio_group[DisplayMode.PLANE_FIXED].buttonClicked.connect(
            self._plane_fixed_mode_switch_modal
        )

        # (5) reset image plane
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if self._radio_btn[DisplayMode.MODAL_FIXED][plane].isChecked():
                for modal in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                    self.img_frame[modal].plane = plane
                break

        # (6) reset img modality
        for modal in [Modal.PT, Modal.MR1, Modal.MR2]:
            if self._radio_btn[DisplayMode.PLANE_FIXED][modal].isChecked():
                for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                    self.img_frame[plane].modal = modal
                break

        # (7) plane fixed mode text labels: "CT" and "other.modality"
        for i in [Modal.CT, "other.modal"]:
            self._text_label[i] = QtWidgets.QLabel()
            self._text_label[i].setFixedHeight(g.TEXT_HEIGHT)
        # set text
        self._text_label[Modal.CT].setText("CT")
        if self._radio_btn[DisplayMode.PLANE_FIXED][Modal.PT].isChecked():
            self._text_label["other.modal"].setText("PT")
        elif self._radio_btn[DisplayMode.PLANE_FIXED][Modal.MR1].isChecked():
            self._text_label["other.modal"].setText("MR-T1")
        elif self._radio_btn[DisplayMode.PLANE_FIXED][Modal.MR2].isChecked():
            self._text_label["other.modal"].setText("MR-T2")
        # alignment
        self._text_label[Modal.CT].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        width = 25 if g.is_linux() else 40
        self._text_label[Modal.CT].setFixedWidth(width)
        self._text_label["other.modal"].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        width = 55 if g.is_linux() else 65
        self._text_label["other.modal"].setFixedWidth(width)

        # ct/pt weight slider bar
        self._slider["mix"] = QtWidgets.QSlider()
        # use TEXT_HEIGHT to make slider has same height as the text labels next to it
        self._slider["mix"].setFixedHeight(g.TEXT_HEIGHT)
        self._slider["mix"].setOrientation(Qt.Horizontal)
        self._slider["mix"].setMinimum(0)
        self._slider["mix"].setMaximum(100)
        self._slider["mix"].setValue(50)
        self._slider["mix"].valueChanged.connect(self.__on_mix_slider_changed)

        # collapse
        self._collap["display.mode"] = QCollapsible("DISPLAY MODE")
        self._collap["display.mode"].expand()
        v_layout = QtWidgets.QVBoxLayout()

        # toggle btn and display mode text
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(self._text_label[DisplayMode.MODAL_FIXED])
        h_layout.addWidget(self._toggle_btn)
        h_layout.addWidget(self._text_label[DisplayMode.PLANE_FIXED])
        v_layout.addLayout(h_layout)

        # modality fixed widgets
        h_layout = QtWidgets.QHBoxLayout()
        for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            h_layout.addWidget(self._radio_btn[DisplayMode.MODAL_FIXED][i])
        v_layout.addLayout(h_layout)

        # plane fixed widgets - mix slider
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(self._text_label[Modal.CT])
        h_layout.addWidget(self._slider["mix"])
        h_layout.addWidget(self._text_label["other.modal"])
        v_layout.addLayout(h_layout)

        # plane fixed widgets
        h_layout = QtWidgets.QHBoxLayout()
        for i in [Modal.PT, Modal.MR1, Modal.MR2]:
            h_layout.addWidget(self._radio_btn[DisplayMode.PLANE_FIXED][i])
        v_layout.addLayout(h_layout)

        # put v_layout into collapsible space
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)
        self._collap["display.mode"].addWidget(container)
        self._collap["display.mode"].expand()

    # this function is connected to widget, dont set input params to this function
    def __on_zoom_slider_changed(self):
        # no need to reload origin_rgb, only start reloading from zoomed_rgb
        self.refresh_imgs(reload_origin_rgb=False)
        self.refresh_crosses()

    def get_zoomin_factor(self):
        return self._slider["zoom"].value() / 100

    def _init_widgets_zoom(self):
        self._slider["zoom"] = QtWidgets.QSlider()
        self._slider["zoom"].setFixedHeight(g.SLIDER_HEIGHT)
        self._slider["zoom"].setOrientation(Qt.Horizontal)
        self._slider["zoom"].setMinimum(100)
        self._slider["zoom"].setMaximum(300)
        self._slider["zoom"].setValue(100)
        self._slider["zoom"].valueChanged.connect(self.__on_zoom_slider_changed)

        # add slider into collapsible space
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(self._slider["zoom"])
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)
        self._collap["zoom"] = QCollapsible("ZOOM IN")
        self._collap["zoom"].addWidget(container)
        self._collap["zoom"].collapse()

    # virtual function (for ui_idl)
    def _init_widgets_todo_list(self):
        return

    # virtual function (for ui_idl)
    def _init_widgets_annotation(self, ui_setting: Dict):
        return

    # virtual function (for ui_idl)
    def _init_widgets_cursor(self):
        return

    def _init_widgets(self, ui_setting: Dict):
        self.__help_msg_box_shown = False

        self._collap = Dict()
        self._radio_btn = Dict()
        self._radio_group = Dict()
        self._text_label = Dict()
        self._slider = Dict()
        self._btn = Dict()

        self._init_widgets_cursor()
        self._init_widgets_todo_list()
        self._init_widgets_combox()
        self._init_widgets_img_frames()
        self._init_widgets_annotation(ui_setting)
        self._init_widgets_display_mode()
        self._init_widgets_color_enhance()
        self._init_widgets_zoom()
        self.switch_display_mode()  # after all widgets initalized

        # set font style
        for i in self._collap.keys():
            self._collap[i].setStyleSheet(g.FONT_STYLE)

        # add collapsible bars into sidebar
        container = QtWidgets.QWidget()
        container.setContentsMargins(0, 0, 0, 0)
        v_layout = QtWidgets.QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(1)
        for i in self._collap.keys():
            v_layout.addWidget(self._collap[i])
        self._side_bar = QtWidgets.QScrollArea(parent=self._central_widget)
        self._side_bar.setWidgetResizable(True)
        self._side_bar.setContentsMargins(0, 0, 0, 0)
        self._side_bar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._side_bar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._side_bar.setWidget(container)

        self.__side_bar_width = 330 if g.is_linux() else 460
        self.setMinimumSize(self.__side_bar_width + 600, 600)
        self.resize(1200, 800)  # set origin size
        self.showMaximized()

    def _refresh_side_bar(self):
        left = 0
        top = 0
        width = self.__side_bar_width - left * 2
        height = self._central_widget.height()
        left += self.geometry().width() - self.__side_bar_width
        self._side_bar.move(left, top)
        self._side_bar.setFixedWidth(width)
        self._side_bar.setFixedHeight(height)

    # this function is connected to widget, dont set input params to this function
    def __on_modal_fixed_radio_group_clicked(self):
        self._modal_fixed_mode_switch_plane()

    def _modal_fixed_mode_switch_plane(
        self,
        new_plane: str = None,  # = None will read from radio buttons
    ):
        # switch plane based on the radio buttons
        if new_plane is None:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                if self._radio_btn[DisplayMode.MODAL_FIXED][plane].isChecked():
                    for modal in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                        self.img_frame[modal].plane = plane
                    break
        # switch to a new plane
        else:
            for modal in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_frame[modal].plane = new_plane
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                if plane == new_plane:
                    self._radio_btn[DisplayMode.MODAL_FIXED][plane].setChecked(True)
                else:
                    self._radio_btn[DisplayMode.MODAL_FIXED][plane].setChecked(False)

        # refresh from origin_rgb as anatomical plane changed
        self.refresh_imgs()
        self.refresh_crosses()

    # this function is connected to widget, dont set input params to this function
    def __switch_color_enhance_slider_bars(self):
        # hide and show sliders
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            if self._radio_btn["color.enhance"][i].isChecked():
                self._slider["bright"][i].show()
                self._slider["contrast"][i].show()
            else:
                self._slider["bright"][i].hide()
                self._slider["contrast"][i].hide()

        # update text label
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            if self._radio_btn["color.enhance"][i].isChecked():
                if i == Modal.CT:
                    key_word = "CT"
                elif i == Modal.PT:
                    key_word = "PT"
                elif i == Modal.MR1:
                    key_word = "MR-T1"
                elif i == Modal.MR2:
                    key_word = "MR-T2"

        self._text_label["bright"].setText("Brightness ({})".format(key_word))
        self._text_label["contrast"].setText("Contrast ({})".format(key_word))

    def _clear_img_frames(self):
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            width = self.img_frame[i].width()
            height = self.img_frame[i].height()
            black_img = np.zeros([width, height, 3])
            qt_image = QtGui.QImage(
                black_img,
                self.img_frame[i].width(),
                self.img_frame[i].height(),
                self.img_frame[i].width() * 3,
                QtGui.QImage.Format_RGB888,
            )
            self.img_frame[i].set_background(qt_image)

    def _enable_arrow_btns(self, combox_name: str):
        # enable/disable prev/next round buttons
        idx = self.combox[combox_name].currentIndex()
        if idx == 0:
            self._arrow_btn["prev.{}".format(combox_name)].setEnabled(False)
        else:
            self._arrow_btn["prev.{}".format(combox_name)].setEnabled(True)

        if idx == (self.combox[combox_name].count() - 1):
            self._arrow_btn["next.{}".format(combox_name)].setEnabled(False)
        else:
            self._arrow_btn["next.{}".format(combox_name)].setEnabled(True)

    def _get_combox_contents(self, combox: CustomComboBox):
        content_list = List()
        for i in range(combox.count()):
            content_list.append(combox.itemText(i))
        return content_list

    def _load_dataset_dir(self):
        # load slice thickness from baseline hyper
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, "baseline")
        fold_dir = Dir.get_sub_dirs(baseline_dir, key_word="fold=", full_path=True)[0]
        baseline_dataset_ver = Json.load(os.path.join(fold_dir, "hyper.json"))[
            "dataset.ver"
        ]

        # set dataset dir based on current patient
        if self._cur_patient in self._patients[DatasetVer.AU]:
            self.dataset_ver = baseline_dataset_ver
        elif self._cur_patient in self._patients[DatasetVer.OBS_STUDY]:
            self.dataset_ver = DatasetVer.OBS_STUDY
        elif self._cur_patient in self._patients[DatasetVer.MDA]:
            self.dataset_ver = DatasetVer.MDA
        else:
            Debug.error_exit("Can't find current patient in testset patients!")

        # set dataset dir and nii spacing
        self._dataset_dir = g.DATASET_DIR[self.dataset_ver]

    def _fill_combox_patient(self):
        # combox_patients = Dir.get_sub_dirs(
        #     os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, "baseline", "patients")
        # )
        # # from "patient=123" to "123"
        # for i in range(len(combox_patients)):
        #     combox_patients[i] = combox_patients[i][len("patient=") :]

        # combox_patients.find_identical_items(self._patients.to_list())
        combox_patients = self._patients.to_list()
        combox_patients.sort()
        self.combox["patient"].addItems(combox_patients)
        self.combox["patient"].setEnabled(True)
        return combox_patients

    # this function is connected to widget, dont set input params to this function
    def _load_baseline_data(self):
        # self._reset_zoomin()
        self.__clear_scores()
        self._clear_img_3d()
        self._clear_img_frames()

        # run this after current text of baseline combox is confirmed
        self._enable_arrow_btns("baseline")

        self._baseline_id = self.combox["baseline"].currentText()

        # reset comboboxes
        for i in ["patient", "idl.gtvt", "idl.gtvn"]:
            self.combox[i].clear()
            self.combox[i].setEnabled(False)
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
        self._load_patient_data(reset_patient)

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

        # flip left/right
        if self.dataset_ver in [DatasetVer.AU]:
            img = np.flip(img, axis=2)

        return img

    # ui_idl will inherit this function, do not make it a private function
    def _load_multi_modal_imgs(self):
        img_path = Dict()
        img_path[Modal.CT] = "CT"
        img_path[Modal.PT] = "PT"
        img_path[Modal.MR1] = "T1dr"
        img_path[Modal.MR2] = "T2dr"
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            img_path[i] = "HNCDL_{}_{}.nii".format(self._cur_patient, img_path[i])
            img_path[i] = os.path.join(self._dataset_dir, img_path[i])
            self.img_3d[i] = self._load_3d_img(img_path[i])

    # this function is connected to widget, dont set input params to this function
    def __on_combox_patient_clicked(self):
        self._load_patient_data()

    def _load_patient_data(self, reset_patient: bool = True):
        # triggered by:
        # (1) patient combox update
        # (2) baseline combox update, but can not find cur patient in new baseline dir
        if reset_patient is True:
            self._cur_patient = self.combox["patient"].currentText()
            # self._reset_zoomin()

        # triggered by baseline combox update, and find cur patient in new baseline dir
        else:
            self.combox["patient"].setCurrentText(self._cur_patient)

        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_dir()

        # reset comboboxes
        for i in ["idl.gtvt", "idl.gtvn"]:
            self.combox[i].clear()
            self.combox[i].setEnabled(False)
            self._arrow_btn["prev.{}".format(i)].setEnabled(False)
            self._arrow_btn["next.{}".format(i)].setEnabled(False)

        # fill idl comboboxes
        for i in ["idl.gtvt", "idl.gtvn"]:
            combox_items = ["baseline"]
            # get all round folder under current patient folder
            for idl_result_dir in Dir.get_sub_dirs(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id),
                key_word=i,
                full_path=True,
            ):
                if Path(idl_result_dir).name == "idl.gtvn_obs.study":
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

            self.combox[i].addItems(combox_items)

            # enable idl.gtvt/gtvn combobox
            self.combox[i].setEnabled(True)
            self._enable_arrow_btns(i)

            # no idl found, show baseline
            if self.combox[i].count() == 1:
                self.combox[i].setCurrentIndex(0)
            # otherwise, show first idl result
            else:
                self.combox[i].setCurrentIndex(1)

        self._load_multi_modal_imgs()
        self.__load_labels()

        # reset slice id (after multi-modal imgs are loaded)
        if self._gtvs_center is not None:
            self.cur_slice_id[Plane.TRANSVERSE] = self._gtvs_center[0]
            self.cur_slice_id[Plane.CORONAL] = self._gtvs_center[1]
            self.cur_slice_id[Plane.SAGITTAL] = self._gtvs_center[2]

        # choose idl automatically
        # try not to reset idl id/round when patient is changed
        for gtv in ["gtvt", "gtvn"]:
            if self._idl_id[gtv] == "baseline":
                reset_id = False
            elif (
                os.path.join(self._idl_id[gtv], self._idl_round[gtv])
                not in self.combox["idl.{}".format(gtv)].currentText()
            ):
                reset_id = True
            else:
                reset_id = False
            # refresh imgs after idl.gtvn is chosen
            self._load_idl_gtv_data(gtv=gtv, reset_id=reset_id, refresh_imgs=False)

        # refresh everything as 3d images are updated
        self.refresh_imgs()

    # load labels and gtvs gravity center
    def __load_labels(self):
        labels = Img.load_labels(
            dataset_dir=self._dataset_dir,
            patient=self._cur_patient,
            nii_load_func=self._load_3d_img,
        )
        # load gtvt and gtvn
        for gtv in ["gtvt", "gtvn"]:
            self.img_3d["{}.label".format(gtv)] = labels[gtv]
        # load gtvs gravity center: (d,h,w)
        self._gtvs_center = list(measurements.center_of_mass(labels["gtvs"]))
        # float to int
        for i in range(len(self._gtvs_center)):
            self._gtvs_center[i] = round(self._gtvs_center[i])

    # this function is connected to widget, dont set input params to this function
    def __on_combox_idl_gtvt_clicked(self):
        self._load_idl_gtvt_data()

    def _load_idl_gtvt_data(self, reset_id: bool = True, refresh_imgs=True):
        self._load_idl_gtv_data(
            gtv="gtvt", reset_id=reset_id, refresh_imgs=refresh_imgs
        )

    # this function is connected to widget, dont set input params to this function
    def __on_combox_idl_gtvn_clicked(self):
        self._load_idl_gtvn_data()

    def _load_idl_gtvn_data(self, reset_id: bool = True, refresh_imgs=True):
        self._load_idl_gtv_data(
            gtv="gtvn", reset_id=reset_id, refresh_imgs=refresh_imgs
        )

    def __clear_gtvt_selected_slices_3d(self):
        self.__gtvt_selected_slices_3d = Dict()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.__gtvt_selected_slices_3d[plane] = List()

    # _load_idl_gtvt_data and _load_idl_gtvn_data will share this function
    def _load_idl_gtv_data(
        self, gtv: str, reset_id: bool = True, refresh_imgs: bool = True
    ):
        # triggered by:
        # (1) idl combox update
        # (2) patient combox update, but can not find cur patient in idl dir
        if reset_id is True:
            combox_item = self.combox["idl.{}".format(gtv)].currentText()
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
                self.combox["idl.{}".format(gtv)].setCurrentText("baseline")
            else:
                self.combox["idl.{}".format(gtv)].setCurrentText(
                    os.path.join(self._idl_id[gtv], self._idl_round[gtv])
                )

        # run this after idl gtvn combox is filled
        self._enable_arrow_btns("idl.{}".format(gtv))

        # load data (pred/clicks/selected_slices)
        # baseline
        if self._idl_id[gtv] == "baseline":
            pred_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "patients",
                "patient={}".format(self._cur_patient),
                "{}_pred.nii.gz".format(gtv),
            )
            # clear idl.gtvt data
            if gtv == "gtvt":
                for i in ["click", "delineation", "correction"]:
                    self.img_3d["gtvt.{}".format(i)] = None
                self.__clear_gtvt_selected_slices_3d()
                # self.__refresh_gtvt_selected_slices_2d()
            # clear idl.gtvn data
            elif gtv == "gtvn":
                for i in ["clicks", "correction"]:
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
            if gtv == "gtvt":
                # load gtvt nii
                for i in ["click", "delineation", "correction"]:
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
                    or self._idl_round["gtvt"] == "round=00"
                ):
                    self.__clear_gtvt_selected_slices_3d()
                else:
                    self.__gtvt_selected_slices_3d = Json.load(
                        selected_slices_json_path
                    )
                    for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                        selected_slices_list = List()
                        for round_num in self.__gtvt_selected_slices_3d[plane]:
                            selected_slices_list += List(
                                self.__gtvt_selected_slices_3d[plane][round_num]
                            )
                            if (round_num) == self._idl_round["gtvt"]:
                                break
                        # str to int
                        for i in range(len(selected_slices_list)):
                            selected_slices_list[i] = int(selected_slices_list[i])
                        self.__gtvt_selected_slices_3d[plane] = selected_slices_list

                # refresh gtvt selected slices (2d)
                # after gtvt selected slices (3d) is loaded
                # self.__refresh_gtvt_selected_slices_2d()

            # load gtvn data
            elif gtv == "gtvn":
                # load gtvn nii
                for i in ["clicks", "correction"]:
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
        if self._idl_id[gtv] == "baseline":
            gtvn_score_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "inference_{}.json".format(self.dataset_ver),
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
                "inference_{}.json".format(self.dataset_ver),
            )
            if os.path.exists(gtvn_score_path):
                gtvn_score = Json.load(gtvn_score_path)
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    self.__scores[gtv][metric] = gtvn_score[
                        "patient={}".format(self._cur_patient)
                    ][metric][self._idl_round[gtv]]

        if refresh_imgs:
            # refresh everything as 3d images are updated
            self.refresh_imgs()

    # this function is connected to widget, dont set input params to this function
    def _load_prev_baseline_data(self):
        idx = self.combox["baseline"].currentIndex() - 1
        if idx < 0:
            return
        prev_baseline = self.combox["baseline"].itemText(idx)
        self.combox["baseline"].setCurrentText(prev_baseline)
        self._load_baseline_data()

    # this function is connected to widget, dont set input params to this function
    def _load_next_baseline_data(self):
        idx = self.combox["baseline"].currentIndex() + 1
        if idx > self.combox["baseline"].count() - 1:
            return
        next_baseline = self.combox["baseline"].itemText(idx)
        self.combox["baseline"].setCurrentText(next_baseline)
        self._load_baseline_data()

    # this function is connected to widget, dont set input params to this function
    def _load_prev_idl_gtvn_data(self):
        idx = self.combox["idl.gtvn"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvn = self.combox["idl.gtvn"].itemText(idx)
        self.combox["idl.gtvn"].setCurrentText(prev_idl_gtvn)
        self._load_idl_gtvn_data()

    # this function is connected to widget, dont set input params to this function
    def _load_next_idl_gtvn_data(self):
        idx = self.combox["idl.gtvn"].currentIndex() + 1
        if idx > self.combox["idl.gtvn"].count() - 1:
            return
        next_idl_gtvn = self.combox["idl.gtvn"].itemText(idx)
        self.combox["idl.gtvn"].setCurrentText(next_idl_gtvn)
        self._load_idl_gtvn_data()

    # this function is connected to widget, dont set input params to this function
    def _load_prev_idl_gtvt_data(self):
        idx = self.combox["idl.gtvt"].currentIndex() - 1
        if idx < 0:
            return
        prev_idl_gtvt = self.combox["idl.gtvt"].itemText(idx)
        self.combox["idl.gtvt"].setCurrentText(prev_idl_gtvt)
        self._load_idl_gtvt_data()

    # this function is connected to widget, dont set input params to this function
    def _load_next_idl_gtvt_data(self):
        idx = self.combox["idl.gtvt"].currentIndex() + 1
        if idx > self.combox["idl.gtvt"].count() - 1:
            return
        next_idl_gtvt = self.combox["idl.gtvt"].itemText(idx)
        self.combox["idl.gtvt"].setCurrentText(next_idl_gtvt)
        self._load_idl_gtvt_data()

    # this function is connected to widget, dont set input params to this function
    def _load_prev_patient_data(self):
        idx = self.combox["patient"].currentIndex() - 1
        if idx < 0:
            return
        prev_patient = self.combox["patient"].itemText(idx)
        self.combox["patient"].setCurrentText(prev_patient)
        self._load_patient_data()

    # this function is connected to widget, dont set input params to this function
    def _load_next_patient_data(self):
        idx = self.combox["patient"].currentIndex() + 1
        if idx > self.combox["patient"].count() - 1:
            return
        next_patient = self.combox["patient"].itemText(idx)
        self.combox["patient"].setCurrentText(next_patient)
        self._load_patient_data()

    # this function is connected to widget, dont set input params to this function
    def __on_mix_slider_changed(self):
        # refresh from origin_rgb as weight of different planes is changed
        self.refresh_imgs()

    def __refresh_imgs_load_origin_rgb(self, frame_name: str):
        modal = self.img_frame[frame_name].modal
        plane = self.img_frame[frame_name].plane
        cur_slice_id = self.cur_slice_id[plane]

        # (1) plane fixed mode
        if self.display_mode() == DisplayMode.PLANE_FIXED:
            if frame_name == Plane.TRANSVERSE:
                slice_ct = self.img_3d[Modal.CT][cur_slice_id, :, :]
                slice_2d = self.img_3d[modal][cur_slice_id, :, :]
            elif frame_name == Plane.CORONAL:
                slice_ct = self.img_3d[Modal.CT][:, cur_slice_id, :]
                slice_2d = self.img_3d[modal][:, cur_slice_id, :]
            elif frame_name == Plane.SAGITTAL:
                slice_ct = self.img_3d[Modal.CT][:, :, cur_slice_id]
                slice_2d = self.img_3d[modal][:, :, cur_slice_id]

            slice_ct = Img.gray_to_rgb(slice_ct)
            if modal == Modal.PT:
                slice_2d = Img.gray_to_colormap(slice_2d)
            else:
                slice_2d = Img.gray_to_rgb(slice_2d)

            # brightness and contrast
            # cv2.addWeighted: dst = src1 * alpha + src2 * beta + gamma
            slice_ct = cv2.addWeighted(
                src1=slice_ct,
                alpha=self._slider["contrast"][Modal.CT].value() / 100,
                src2=np.zeros_like(slice_ct),
                beta=0,
                gamma=self._slider["bright"][Modal.CT].value(),
            )
            slice_2d = cv2.addWeighted(
                src1=slice_2d,
                alpha=self._slider["contrast"][modal].value() / 100,
                src2=np.zeros_like(slice_2d),
                beta=0,
                gamma=self._slider["bright"][modal].value(),
            )

            # mix ct and the other modality
            alpha = self._slider["mix"].value() / 100
            origin_rgb = cv2.addWeighted(
                src1=slice_2d,
                alpha=alpha,
                src2=slice_ct,
                beta=1 - alpha,
                gamma=0,
            )

        # (2) modality fixed mode
        else:
            if plane == Plane.TRANSVERSE:
                slice_2d = self.img_3d[frame_name][cur_slice_id, :, :]
            elif plane == Plane.CORONAL:
                slice_2d = self.img_3d[frame_name][:, cur_slice_id, :]
            elif plane == Plane.SAGITTAL:
                slice_2d = self.img_3d[frame_name][:, :, cur_slice_id]
            slice_2d = Img.gray_to_rgb(slice_2d)

            # brightness and contrast
            # cv2.addWeighted: dst = src1 * alpha + src2 * beta + gamma
            origin_rgb = cv2.addWeighted(
                src1=slice_2d,
                alpha=self._slider["contrast"][modal].value() / 100,
                src2=np.zeros_like(slice_2d),
                beta=0,
                gamma=self._slider["bright"][modal].value(),
            )

        self._origin_rgb[frame_name] = origin_rgb

    def __refresh_imgs_add_contours(self, frame_name: str):
        # make sure contoured_rgb is not None,
        # because sometime there is not contour to draw
        self._contoured_rgb[frame_name] = self._zoomed_rgb[frame_name].copy()

        if not self.__show_contour:
            return

        plane = self.img_frame[frame_name].plane

        # idl mode
        # "delete_all_crosses" is unique a function belonging to ObsStudyWindow
        # for hasattr() function has to be a public or protected one, not private
        if hasattr(self, "delete_all_crosses"):
            # place top contour at the end of the list, click > pred.final
            seg_name_list = [
                "gtvn.pred.final",
                "gtvt.pred.final",
                "gtvt.delineation",
                "gtvn.clicks",
                "gtvt.click",
            ]
        # replay mode
        else:
            # place top contour at the end of the list
            # click > correction > delineation > pred > label
            seg_name_list = [
                "gtvn.label",
                "gtvt.label",
                "gtvn.pred",
                "gtvt.pred",
                "gtvt.delineation",
                "gtvn.correction",
                "gtvt.correction",
                "gtvn.clicks",
                "gtvt.click",
            ]

        # loop through segmentation 3d imgs
        for seg_name in seg_name_list:
            if self.img_3d[seg_name] is None:
                continue

            # load data of current slice
            if plane == Plane.SAGITTAL:
                segment = self.img_3d[seg_name][:, :, self.cur_slice_id[Plane.SAGITTAL]]
            elif plane == Plane.CORONAL:
                segment = self.img_3d[seg_name][:, self.cur_slice_id[Plane.CORONAL], :]
            elif plane == Plane.TRANSVERSE:
                segment = self.img_3d[seg_name][
                    self.cur_slice_id[Plane.TRANSVERSE], :, :
                ]
            segment = segment.astype(np.uint8)

            # skip if current segmentation is empty
            if seg_name in [
                "gtvn.pred.final",
                "gtvt.pred.final",
                "gtvn.correction",
                "gtvt.correction",
                "gtvt.delineation",
            ]:
                # perfomr erosion to remove overlap of 3 different planes
                kernel = np.ones((3, 3), np.uint8)
                eroded_segment = cv2.erode(segment, kernel, iterations=1)
                if eroded_segment.max() <= 0:
                    continue
            else:
                if segment.max() <= 0:
                    continue

            # zoom in segmentation
            zoomed_h = self._zoomed_rgb[frame_name].shape[0]
            zoomed_w = self._zoomed_rgb[frame_name].shape[1]
            segment = cv2.resize(
                segment,
                (zoomed_w, zoomed_h),
                interpolation=cv2.INTER_AREA,
            )

            # use higher thickness for click, otherwise cant see the points
            if seg_name == "gtvt.click" or seg_name == "gtvn.clicks":
                thickness = 7
            # use lower thickness for contours
            else:
                thickness = 2
                # GaussianBlur after zoomed in
                kernel_size = (7, 7)
                segment = cv2.GaussianBlur(segment, kernel_size, cv2.BORDER_DEFAULT)

            # draw contour on zoomed in rgb
            contours = cv2.findContours(
                segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            self._contoured_rgb[frame_name] = cv2.drawContours(
                image=self._contoured_rgb[frame_name],
                contours=contours,
                contourIdx=-1,
                color=self.color[seg_name],
                thickness=thickness,
            )

    def __refresh_imgs_load_zoomed_rgb(self, frame_name: str):
        origin_h = self._origin_rgb[frame_name].shape[0]
        origin_w = self._origin_rgb[frame_name].shape[1]
        frame_w = self.img_frame[frame_name].width()
        frame_h = self.img_frame[frame_name].height()

        # img is aligned to top and bottom of img_frame
        if origin_h * frame_w > frame_h * origin_w:
            zoomed_w = round(frame_h * origin_w / origin_h)
            zoomed_h = frame_h
        # img is aligned to left and right of img_frame
        else:
            zoomed_w = frame_w
            zoomed_h = round(frame_w * origin_h / origin_w)

        zoomed_w *= self.get_zoomin_factor()
        zoomed_w = round(zoomed_w)
        zoomed_h *= self.get_zoomin_factor()
        zoomed_h = round(zoomed_h)

        self._zoomed_rgb[frame_name] = cv2.resize(
            self._origin_rgb[frame_name],
            (zoomed_w, zoomed_h),
            interpolation=cv2.INTER_AREA,
        )

        # GaussianBlur after zoomed in looks better
        self._zoomed_rgb[frame_name] = cv2.GaussianBlur(
            self._zoomed_rgb[frame_name], (3, 3), cv2.BORDER_DEFAULT
        )

    def __refresh_imgs_fill_img_frame(
        self,
        frame_name: str,
        img_pos_diff: tuple = (0, 0),  # Default to (0, 0) if not provided
    ):

        img_center_pct = self.img_frame[frame_name].img_center_pct
        frame_w = self.img_frame[frame_name].width()
        frame_h = self.img_frame[frame_name].height()
        zoomed_h = self._contoured_rgb[frame_name].shape[0]
        zoomed_w = self._contoured_rgb[frame_name].shape[1]

        # Calculate the absolute center position and adjust with img_pos_diff
        center_x_abs = round(zoomed_w * img_center_pct[0]) - img_pos_diff[0]
        center_y_abs = round(zoomed_h * img_center_pct[1]) - img_pos_diff[1]

        # Ensure the center coordinates are within the image bounds after adjustment
        center_x_abs = max(0, min(zoomed_w, center_x_abs))
        center_y_abs = max(0, min(zoomed_h, center_y_abs))

        # Determine padding or cropping for width to align the specified position
        if zoomed_w <= frame_w:
            total_padding_w = frame_w - zoomed_w
            # Distribute padding evenly to ensure centering
            pad_width_left = total_padding_w // 2
            pad_width_right = total_padding_w - pad_width_left
            final_rgb = np.pad(
                self._contoured_rgb[frame_name],
                [(0, 0), (pad_width_left, pad_width_right), (0, 0)],
                mode="constant",
            )
            # update center pct
            center_x_pct = 0.5
        else:
            start_x = max(0, min(zoomed_w - frame_w, center_x_abs - frame_w // 2))
            final_rgb = self._contoured_rgb[frame_name][
                :, start_x : start_x + frame_w, :
            ]
            # update center pct
            center_x_abs = max(frame_w // 2, center_x_abs)
            center_x_abs = min(zoomed_w - frame_w // 2, center_x_abs)
            center_x_pct = center_x_abs / zoomed_w

        # Determine padding or cropping for height after width adjustment
        if zoomed_h <= frame_h:
            total_padding_h = frame_h - zoomed_h
            # Distribute padding evenly to ensure centering
            pad_height_top = total_padding_h // 2
            pad_height_bottom = total_padding_h - pad_height_top
            final_rgb = np.pad(
                final_rgb,
                [(pad_height_top, pad_height_bottom), (0, 0), (0, 0)],
                mode="constant",
            )
            # update center pct
            center_y_pct = 0.5
        else:
            start_y = max(0, min(zoomed_h - frame_h, center_y_abs - frame_h // 2))
            final_rgb = final_rgb[start_y : start_y + frame_h, :, :]
            # update center pct
            center_y_abs = max(frame_h // 2, center_y_abs)
            center_y_abs = min(zoomed_h - frame_h // 2, center_y_abs)
            center_y_pct = center_y_abs / zoomed_h

        # img_center_pct
        self.img_frame[frame_name].img_center_pct = (center_x_pct, center_y_pct)

        return final_rgb

    def refresh_imgs(
        self,
        frame_name: str = None,
        reload_origin_rgb: bool = True,
        reload_zoomed_rgb: bool = True,
        reload_contours: bool = True,
        img_pos_diff: tuple = (0, 0),  # this is for right click drag and move
    ):
        # no patient loaded (no img_3d loaded)
        if self.img_3d[Modal.CT] is None:
            if self.display_mode() == DisplayMode.PLANE_FIXED:
                frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
            else:
                frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
            for frame_name in frame_name_list:
                w = self.img_frame[frame_name].width()
                h = self.img_frame[frame_name].height()
                qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB888)
                black = QtGui.QColor(0, 0, 0)
                qimg.fill(black)

                # add msg on qimg: "please select a patient"
                if frame_name == Plane.TRANSVERSE or frame_name == Modal.CT:
                    self._add_instruction_on_top_left(qimg)

                self.img_frame[frame_name].set_background(qimg)
                self.img_frame[frame_name].update()
            return

        # img name
        if frame_name is not None:
            frame_name_list = [frame_name]
        else:
            if self.display_mode() == DisplayMode.PLANE_FIXED:
                frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
            else:
                frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]

        # load rgb imgs
        for frame_name in frame_name_list:
            if reload_origin_rgb or self._origin_rgb[frame_name] is None:
                self.__refresh_imgs_load_origin_rgb(frame_name)

            if reload_zoomed_rgb or self._zoomed_rgb[frame_name] is None:
                self.__refresh_imgs_load_zoomed_rgb(frame_name)

            if reload_contours or self._contoured_rgb[frame_name] is None:
                self.__refresh_imgs_add_contours(frame_name)

            # resize and fit img qlabel
            final_rgb = self.__refresh_imgs_fill_img_frame(
                frame_name=frame_name,
                img_pos_diff=img_pos_diff,
            )

            # avoid Non-Contiguity problem
            # this happens when img.width/height are all larger than img_frame
            # and crop image will cause Non-Contiguity problem
            if not final_rgb.flags["C_CONTIGUOUS"]:
                # make ndarray C-contiguous
                final_rgb = np.ascontiguousarray(final_rgb)

            # ndarray to qimage
            rgb_height = final_rgb.shape[0]
            rgb_width = final_rgb.shape[1]
            rgb_chan = final_rgb.shape[2]
            qimg = QtGui.QImage(
                final_rgb,  # data:bytes
                rgb_width,
                rgb_height,
                rgb_width * rgb_chan,  # bytesPerLine
                QtGui.QImage.Format_RGB888,  # format
            )

            # add text on top left
            if frame_name == Plane.TRANSVERSE or frame_name == Modal.CT:
                self._add_score_on_top_left(qimg)
                self._add_instruction_on_top_left(qimg)

            # add contour description on bottom left
            if frame_name == Plane.TRANSVERSE or frame_name == Modal.MR1:
                self._add_contour_description_on_bottom_left(qimg)

            # add slice number on bottom right
            self.__add_slice_id_on_bottom_right(frame_name=frame_name, qimg=qimg)

            self.img_frame[frame_name].set_background(qimg)
            self.img_frame[frame_name].update()

    def __add_slice_id_on_bottom_right(self, frame_name: str, qimg: QtGui.QImage):
        if self.img_3d[Modal.CT] is None:
            return

        plane = self.img_frame[frame_name].plane

        cur_slice_id = self.cur_slice_id[plane] + 1
        if plane == Plane.SAGITTAL:
            slice_count = self.img_3d[Modal.CT].shape[2]
        elif plane == Plane.CORONAL:
            slice_count = self.img_3d[Modal.CT].shape[1]
        elif plane == Plane.TRANSVERSE:
            slice_count = self.img_3d[Modal.CT].shape[0]
        text = "{:03d}/{:03d}".format(cur_slice_id, slice_count)

        left = qimg.width() - 83
        bottom = self._get_text_pos_bottom(qimg)[0]
        self._qimg_draw_text(
            qimg=qimg,
            text=text,
            pos=(left, bottom),
            color=self.color["green"],
        )

    def _get_text_pos_top(self):
        return 25 if g.is_linux() else 28

    def _get_text_pos_left(self):
        return [10, 65, 110]

    # from buttom to top
    def _get_text_pos_bottom(self, qimg: QtGui.QImage):
        buttom = qimg.height() - 13
        step = 22 if g.is_linux() else 25
        return [
            buttom,
            buttom - step,
            buttom - step * 2,
        ]

    def _add_contour_description_on_bottom_left(self, qimg: QtGui.QImage):
        left = self._get_text_pos_left()
        bottom = self._get_text_pos_bottom(qimg)

        # label
        self._qimg_draw_text(
            qimg=qimg,
            text="Label:",
            pos=(left[0], bottom[2]),
            color=self.color["green"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVt",
            pos=(left[1], bottom[2]),
            color=self.color["gtvt.label"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVn",
            pos=(left[2], bottom[2]),
            color=self.color["gtvn.label"],
        )

        # pred
        self._qimg_draw_text(
            qimg=qimg,
            text="Pred:",
            pos=(left[0], bottom[1]),
            color=self.color["green"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVt",
            pos=(left[1], bottom[1]),
            color=self.color["gtvt.pred"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="GTVn",
            pos=(left[2], bottom[1]),
            color=self.color["gtvn.pred"],
        )

        # user input
        self._qimg_draw_text(
            qimg=qimg,
            text="User:",
            pos=(left[0], bottom[0]),
            color=self.color["green"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="Init",
            pos=(left[1], bottom[0]),
            color=self.color["gtvt.delineation"],
        )
        self._qimg_draw_text(
            qimg=qimg,
            text="Correction",
            pos=(left[2], bottom[0]),
            color=self.color["gtvt.correction"],
        )

    def _add_instruction_on_top_left(self, qimg: QtGui.QImage):
        pass

    def _add_score_on_top_left(self, qimg: QtGui.QImage):
        top = self._get_text_pos_top()

        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            left = self._get_text_pos_left()[0]

            # "Metric.DSC/Metric.MSD/Metric.HD95: "
            text = metric.upper() + ": "
            self._qimg_draw_text(
                qimg=qimg,
                text=text,
                pos=(left, top),
                color=self.color["green"],
            )
            # load scores
            for i in ["gtvt", "gtvn"]:
                # text
                if Value.is_number(self.__scores[i][metric]):
                    if metric == Metric.DSC:
                        text = "{:.2f}".format(self.__scores[i][metric])
                    else:
                        text = "{:.1f}".format(self.__scores[i][metric])
                else:
                    text = "NaN"
                # mod x pos
                if i == "gtvt":
                    left += 55
                else:
                    left += 50
                # draw text
                self._qimg_draw_text(
                    qimg=qimg,
                    text=text,
                    pos=(left, top),
                    color=self.color["{}.pred".format(i)],
                )
            # mod y pos
            top += 20

    # abstract function for ObsStudyWindow, triggerd by ImgFrame.enterEvent()
    def change_mouse_cursor(self, check_mouse_over_img_frame: bool):
        return

    # abstract function for ObsStudyWindow, triggerd by ImgFrame.enterEvent()
    def restore_mouse_cursor(self):
        return

    def _qimg_draw_text(
        self,
        qimg,
        text: str,
        pos: tuple,
        color: tuple,
        line_gap: int = 20,
    ):
        font = QtGui.QFont("Arial", 12)
        font.setBold(True)
        painter = QtGui.QPainter(qimg)
        painter.setFont(font)
        r, g, b = color
        alpha = 255

        x = pos[0]
        for i, line in enumerate(text.split("\n")):
            y = pos[1] + i * line_gap

            # draw outline
            # outline_color = QtGui.QColor("black")
            painter.setPen(Qt.black)
            # Adjust for desired thickness
            offsets = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
            for x_off, y_off in offsets:
                painter.drawText(x + x_off, y + y_off, line)

            # draw text
            painter.setPen(QtGui.QColor(r, g, b, alpha))
            painter.drawText(x, y, line)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__resize_img_frames()
        self._refresh_side_bar()
        # no need to refresh origin_rgb
        # refresh from zoomed_rgb as img frame size is changed
        self.refresh_imgs(reload_origin_rgb=False)
        self.refresh_crosses()

    def __resize_img_frames(self):
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
        self.img_frame[Modal.CT].setGeometry(
            QtCore.QRect(pos["x"][0], pos["y"][0], pos["w"][0], pos["h"][0])
        )
        # transverse
        self.img_frame[Plane.TRANSVERSE].setGeometry(
            QtCore.QRect(
                pos["x"][0], pos["y"][0], pos["w"][0], pos["h"][0] + 1 + pos["h"][1]
            )
        )
        # pt and coronal
        for i in [Modal.PT, Plane.CORONAL]:
            self.img_frame[i].setGeometry(
                QtCore.QRect(pos["x"][1], pos["y"][0], pos["w"][1], pos["h"][0])
            )
        # mr1
        self.img_frame[Modal.MR1].setGeometry(
            QtCore.QRect(pos["x"][0], pos["y"][1], pos["w"][0], pos["h"][1])
        )
        # mr2 and sagittal
        for i in [Modal.MR2, Plane.SAGITTAL]:
            self.img_frame[i].setGeometry(
                QtCore.QRect(pos["x"][1], pos["y"][1], pos["w"][1], pos["h"][1])
            )

    def _open_file_dlg(self):
        Tk().withdraw()
        file_name = filedialog.askopenfilename()
        if file_name == "" or file_name is None:
            pass

    def _check_focus(self):
        focused_widget = QtWidgets.QApplication.focusWidget()
        if focused_widget:
            print("Current focus:", focused_widget.objectName())
        else:
            print("No focus at the moment.")

    def __event_key_space(self):
        # image not loaded
        if self.img_3d[Modal.CT] is None:
            return
        # only switch the mix slider in PLANE_FIXED mode
        if self.display_mode() == DisplayMode.PLANE_FIXED:
            if self._slider["mix"].value() >= 50:
                new_val = self._slider["mix"].minimum()
                self._slider["mix"].setValue(new_val)
            else:
                new_val = self._slider["mix"].maximum()
                self._slider["mix"].setValue(new_val)

    def __event_key_page_up_down(self, event):
        # image not loaded
        if self.img_3d[Modal.CT] is None:
            return

        # check which img is under mouse
        img_frame_name = self._under_mouse_img_frame_name()
        if img_frame_name is None:
            return

        plane = self.img_frame[img_frame_name].plane
        if plane == Plane.SAGITTAL:
            slice_count = self.img_3d[Modal.CT].shape[2]
        elif plane == Plane.CORONAL:
            slice_count = self.img_3d[Modal.CT].shape[1]
        elif plane == Plane.TRANSVERSE:
            slice_count = self.img_3d[Modal.CT].shape[0]

        if event.key() == Qt.Key_PageUp:
            self.cur_slice_id[plane] -= 1
        elif event.key() == Qt.Key_PageDown:
            self.cur_slice_id[plane] += 1
        # limite slice_id in range (0, slice_count)
        self.cur_slice_id[plane] %= slice_count

        # refresh imgs: refresh everything for a new slice
        # (1) PLANE_FIXED mode, only refresh current img frame
        if self.display_mode() == DisplayMode.PLANE_FIXED:
            self.refresh_imgs(frame_name=img_frame_name)
        # (2) MODAL_FIXED mode, refresh all 4 img frames
        else:
            self.refresh_imgs()

    def __event_key_f1(self):
        if not self.__help_msg_box_shown:
            text = (
                "Software Name: Interactive Deep-learning Tool\n"
                "\n"
                "Mouse wheel up - Previous slice\n"
                "Mouse wheel down - Next slice.\n"
                "Ctrl + mouse wheel up - Zoom in.\n"
                "Ctrl + mouse wheel down - Zoom out.\n"
                "I - Zoom in.\n"
                "O - Zoom out.\n"
                "X - Show/Hide contours.\n"
                "Left click - Paint.\n"
                "Right click - Drag and move image (when zoomed in).\n"
                "Delete/Backspace - Remove a selected GTVt/GTVn center.\n"
            )
            self.__help_msg_box_shown = True
            QMessageBox.information(
                self,
                "Help",
                text,
                QMessageBox.Ok,
            )
            self.__help_msg_box_shown = False

    def eventFilter(self, source, event):
        # Check if the event is a key press event
        if event.type() == QtCore.QEvent.KeyPress:
            # spacebar pressed
            if event.key() == Qt.Key_Space:
                self.__event_key_space()

            # pageup or pagedown pressed, goto prev/next slice
            elif event.key() == Qt.Key_PageUp or event.key() == Qt.Key_PageDown:
                self.__event_key_page_up_down(event)

            # press "I"
            elif event.key() == Qt.Key_I:
                self.__zoom_in(40)

            # press "O"
            elif event.key() == Qt.Key_O:
                self.__zoom_out(40)

            # press "X"
            elif event.key() == Qt.Key_X:
                if self.__show_contour:
                    self.__show_contour = False
                else:
                    self.__show_contour = True
                # only refresh contour
                self.refresh_imgs(
                    reload_origin_rgb=False,
                    reload_zoomed_rgb=False,
                )

            # press "F1"
            elif event.key() == Qt.Key_F1:
                self.__event_key_f1()

            elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                self.delete_selected_crosses()

            # return True means event is handled
            # block all key press event from reaching any widget
            return True

        # use eventFilter to handle Ctrl+Wheel events for the parent
        # otherwise only child widget's wheel event is triggered
        elif event.type() == QtCore.QEvent.Wheel:
            # hide popup of all combobox
            for i in self.combox.keys():
                self.combox[i].hidePopup()

            img_frame_name = self._under_mouse_img_frame_name()

            if img_frame_name is not None:
                # ctrl + wheel up/down
                if QtWidgets.QApplication.keyboardModifiers() == Qt.ControlModifier:
                    # wheel up, zoom in
                    if event.angleDelta().y() > 0:
                        self.__zoom_in(step=20)
                    # wheel down, zoom out
                    else:
                        self.__zoom_out(step=20)

                # wheel up/down (without ctrl)
                else:
                    self.__event_wheel_switch_new_slice(
                        event=event, img_frame_name=img_frame_name
                    )

            elif self._side_bar.underMouse():
                self.__event_wheel_on_side_bar(event)

            # return True means event is handled
            # block all other wheel event from reaching any widget
            return True

        # For other events, call the base class method to ensure standard event processing
        else:
            return super().eventFilter(source, event)

    def __event_wheel_on_side_bar(self, event: QtCore.QEvent):
        # control vertical scroll bar
        v_scroll_bar = self._side_bar.verticalScrollBar()
        if event.angleDelta().y() > 0:
            v_scroll_bar.setValue(v_scroll_bar.value() - 20)
        else:
            v_scroll_bar.setValue(v_scroll_bar.value() + 20)

    # abstract function for ObsStudyWindow
    def delete_selected_crosses(self):
        return

    def __event_wheel_switch_new_slice(self, event: QtCore.QEvent, img_frame_name: str):

        ct_img = self.img_3d[Modal.CT]
        if ct_img is None:
            return

        plane = self.img_frame[img_frame_name].plane

        if plane == Plane.SAGITTAL:
            slice_count = ct_img.shape[2]
        elif plane == Plane.CORONAL:
            slice_count = ct_img.shape[1]
        elif plane == Plane.TRANSVERSE:
            slice_count = ct_img.shape[0]

        if event.angleDelta().y() > 0:
            slice_delta = 1
        else:
            slice_delta = -1
        if plane == Plane.CORONAL:
            slice_delta = -slice_delta
        elif plane == Plane.TRANSVERSE:
            slice_delta *= self.interpolation_step

        # update slice id
        new_slice_id = self.cur_slice_id[plane] - slice_delta
        # make slice_id cycle in [0, slice_count-1]
        if new_slice_id > slice_count - 1:
            new_slice_id = 0
        elif new_slice_id < 0:
            new_slice_id = slice_count - 1
        # make sure transverse slice id is a multiple of interpolation step
        if plane == Plane.TRANSVERSE:
            new_slice_id = self.ensure_slice_id_multiple(
                slice_id=new_slice_id,
                slice_count=slice_count,
            )
        self.cur_slice_id[plane] = new_slice_id

        # refresh new slice
        # (1) PLANE_FIXED mode, only refresh current img frame
        if self.display_mode() == DisplayMode.PLANE_FIXED:
            self.refresh_imgs(frame_name=plane)
            self.refresh_crosses(frame_name=plane)
        # (2) MODAL_FIXED mode, refresh all 4 img frames
        else:
            self.refresh_imgs()
            self.refresh_crosses()

    def _under_mouse_img_frame_name(self) -> str:
        for i in self.img_frame.keys():
            if self.img_frame[i].underMouse():
                return i
        return None

    def __zoom_in(self, step: int):
        cur_value = self._slider["zoom"].value()
        max_value = self._slider["zoom"].maximum()
        if cur_value >= max_value:
            return
        else:
            new_value = min(cur_value + step, max_value)
            self._slider["zoom"].setValue(new_value)
            # reload from zoomed_rgb, no need to reload origin_rgb
            self.refresh_imgs(reload_origin_rgb=False)
            self.refresh_crosses()

    def __zoom_out(self, step: int):
        cur_value = self._slider["zoom"].value()
        min_value = self._slider["zoom"].minimum()
        if cur_value <= min_value:
            return
        else:
            new_value = max(cur_value - step, min_value)
            self._slider["zoom"].setValue(new_value)
            # reload from zoomed_rgb, no need to reload origin_rgb
            self.refresh_imgs(reload_origin_rgb=False)
            self.refresh_crosses()
