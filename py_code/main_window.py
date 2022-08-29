import global_elems as g
import SimpleITK as sitk
import os
import sys
import cv2
import numpy as np
from nested_dict import NestedDict
from datetime import datetime
from tkinter import Tk
from tkinter import filedialog
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QRect, Qt, QSize
from PyQt5.QtGui import QPalette, QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLayout,
    QMainWindow,
    QMenu,
    QAction,
    QRubberBand,
)
from Ui_main_window import Ui_MainWindow

# opencv.shape（height, width，channel）


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._status_bar.hide()
        self._menu_bar.hide()

        self.__init_ui_names()
        self.__init_zoomin()
        self.__init_color()

        self.__init_img_data()
        self.__refresh_title()  # after __init_img_data

        # comes after self.__init_img_data(), because function connection needed
        self.__init_side_bar()

        # resize
        self.resize(1200, 800)  # set origin size
        self.showMaximized()
        # self.setWindowState(Qt.WindowMaximized)

    def __init_zoomin(self):
        self.__zoomin = NestedDict()
        self.__zoomin["rubber_band"] = QRubberBand(QRubberBand.Rectangle, self)
        self.__reset_zoomin()

    def __reset_zoomin(self):
        self.__zoomin["rubber_band"].hide()
        self.__zoomin["img"] = None
        self.__zoomin["start"] = None
        self.__zoomin["end"] = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        # loop 4 img frames
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            left = self.__img_frames[i].x()
            top = self.__img_frames[i].y()
            width = self.__img_frames[i].width()
            height = self.__img_frames[i].height()
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
                    self.__zoomin["rubber_band"].setGeometry(rect.normalized())
                    self.__zoomin["rubber_band"].show()
                    return

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.__zoomin["start"] is None:  # or self.__zoomin["rubber_band"] is None:
            return
        self.__mouse_move_event(event)

    def __mouse_move_event(self, event):
        # limit zoomin frame in img frame
        img_frame = self.__img_frames[self.__zoomin["img"]]
        img_frame_right = img_frame.x() + img_frame.width() - 1
        if event.x() < img_frame.x():
            event_x = img_frame.x()
        elif event.x() > img_frame_right:
            event_x = img_frame_right
        else:
            event_x = event.x()
        img_frame_buttom = img_frame.y() + img_frame.height() - 1
        if event.y() < img_frame.y():
            event_y = img_frame.y()
        elif event.y() > img_frame_buttom:
            event_y = img_frame_buttom
        else:
            event_y = event.y()
        # resize zoomin frame
        self.__zoomin["end"] = QPoint(event_x, event_y)
        rect = QRect(
            self.__zoomin["start"],
            self.__zoomin["end"],
        )
        self.__zoomin["rubber_band"].setGeometry(rect.normalized())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        # not zoomed in
        if self.__zoomin["start"] is None:  # or self.__zoomin["rubber_band"] is None:
            return
        self.__mouse_move_event(event)
        self.__zoomin["rubber_band"].hide()
        # self.__zoomin["rubber_band"] = None

        # no data loaded
        if self.__img_data["ct"] is None:
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
        # get img_frame related position
        img_frame_left = self.__img_frames[self.__zoomin["img"]].x()
        img_frame_top = self.__img_frames[self.__zoomin["img"]].y()
        start_x -= img_frame_left
        end_x -= img_frame_left
        start_y -= img_frame_top
        end_y -= img_frame_top

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
            origin_width = self.__img_data["ct"].shape[1]
            origin_height = self.__img_data["ct"].shape[0]
            origin_height = round(origin_height * g.NII_SPACING[2] / g.NII_SPACING[1])
        elif self.__img_plane == "coronal":
            origin_width = self.__img_data["ct"].shape[2]
            origin_height = self.__img_data["ct"].shape[0]
            origin_height = round(origin_height * g.NII_SPACING[2] / g.NII_SPACING[0])
        else:
            origin_width = self.__img_data["ct"].shape[2]
            origin_height = self.__img_data["ct"].shape[1]

        start_x = round(start_x * origin_width / resize_pos["width"])
        end_x = round(end_x * origin_width / resize_pos["width"])
        start_y = round(start_y * origin_height / resize_pos["height"])
        end_y = round(end_y * origin_height / resize_pos["height"])

        self.__zoomin["start"] = QPoint(start_x, start_y)
        self.__zoomin["end"] = QPoint(end_x, end_y)
        self.__refresh_imgs()

    def __fit_img_frame(self, img, img_frame: QtWidgets.QLabel):
        err_msg = "MainWindow.__fit_img_frame(), img.shape should == 2 or 3"

        # image spacing resize
        if self.__img_plane == "sagittal":
            spacing_height = round(img.shape[0] * g.NII_SPACING[2] / g.NII_SPACING[1])
            img = cv2.resize(
                img,
                (
                    img.shape[1],
                    spacing_height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        elif self.__img_plane == "coronal":
            spacing_height = round(img.shape[0] * g.NII_SPACING[2] / g.NII_SPACING[0])
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
                g.exit_app(err_msg)

        # resize to fit image frame
        origin_height = img.shape[0]
        origin_width = img.shape[1]
        resize_pos = NestedDict()
        resize_pos["x"], resize_pos["y"] = None, None
        resize_pos["width"], resize_pos["height"] = None, None
        final_width = img_frame.width()
        final_height = img_frame.height()

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
                g.exit_app(err_msg)
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
                g.exit_app(err_msg)
            img = cv2.resize(
                img,
                (resize_pos["width"], resize_pos["height"]),
                interpolation=cv2.INTER_AREA,
            )
            img = np.concatenate((black_border, img, black_border), axis=0)

        # smooth img
        return img, resize_pos

    def __init_color(self):
        self.__color = NestedDict()
        self.__color["label"] = (0, 255, 255)  # light blue
        self.__color["pred"] = (255, 255, 0)  # yellow
        self.__color["annotated"] = (0, 255, 64)  # green
        self.__color["score.text"] = (0, 255, 64)  # green

    def __init_ui_names(self):
        self.__img_frames = NestedDict()
        self.__img_frames["ct"] = self._img_frame_ct
        self.__img_frames["pt"] = self._img_frame_pt
        self.__img_frames["mrt1"] = self._img_frame_mrt1
        self.__img_frames["mrt2"] = self._img_frame_mrt2

        self.__text_labels = NestedDict()
        self.__text_labels["train.id"] = self._text_label_idl_id
        self.__text_labels["patient"] = self._text_label_patient
        self.__text_labels["round"] = self._text_label_round
        self.__text_labels["bright"] = self._text_label_bright
        self.__text_labels["contrast"] = self._text_label_contrast

        self.__comboxes = NestedDict()
        self.__comboxes["train.id"] = self._combox_idl_id
        self.__comboxes["patient"] = self._combox_patient
        self.__comboxes["round"] = self._combox_round

        self.__radio_btns = NestedDict()
        self.__radio_btns["transverse"] = self._radio_btn_transverse
        self.__radio_btns["coronal"] = self._radio_btn_coronal
        self.__radio_btns["sagittal"] = self._radio_btn_sagittal

        self.__sliders = NestedDict()
        self.__sliders["bright"] = self._slider_bright
        self.__sliders["contrast"] = self._slider_contrast

        # set label background black
        pal = QPalette()
        pal.setColor(QPalette.Window, Qt.black)
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__img_frames[i].setObjectName("")
            self.__img_frames[i].setAutoFillBackground(True)
            self.__img_frames[i].setPalette(pal)

    def __load_img(self, img_path: str):
        img = g.load_nii(img_path)
        img[img < 0] = 0
        img[img > 255] = 255
        # turn upside down
        img = np.flip(m=img, axis=0)
        # img = np.rot90(m=img, k=2, axes=(0, 1))
        return img

    def __init_img_data(self):
        self.__patient = None
        self.__round = None
        self.__slice_id = None  # starts from 0
        self.__score = NestedDict()
        self.__img_data = NestedDict()
        self.__resize_pos = NestedDict()
        self.__clear_img_data()

    def __clear_img_data(self):
        self.__idl_id = None
        self.__score["dsc"] = None
        self.__score["msd"] = None
        self.__score["hd95"] = None

        for i in ["ct", "pt", "mrt1", "mrt2", "label", "pred"]:
            self.__img_data[i] = None

        # resize position of ct/pt/mr1/mr2
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            self.__resize_pos[i] = None

        # transverse/coronal/sagittal
        for i in ["transverse", "coronal", "sagittal"]:
            if self.__radio_btns[i].isChecked():
                self.__img_plane = i

    def __init_side_bar(self):
        # set text
        self.__text_labels["train.id"].setText("Choose iDL Result")
        self.__text_labels["patient"].setText("Choose Patient")
        self.__text_labels["round"].setText("Choose Update Round")
        self.__text_labels["bright"].setText("Brightness")
        self.__text_labels["contrast"].setText("Contrast")
        self.__radio_btns["transverse"].setText("Transverse")
        self.__radio_btns["coronal"].setText("Coronal")
        self.__radio_btns["sagittal"].setText("Sagittal")

        # set font
        font = self.__text_labels["train.id"].font()
        font.setPointSize(8)
        font.setBold(True)
        for i in ["train.id", "patient", "round", "bright", "contrast"]:
            self.__text_labels[i].setFont(font)
        for i in ["transverse", "coronal", "sagittal"]:
            self.__radio_btns[i].setFont(font)
        font.setBold(False)
        for i in ["train.id", "patient", "round"]:
            self.__comboxes[i].setFont(font)

        # connect widget to function
        idl_results_folders = g.get_sub_folders(g.IDL_RESULTS_FOLDER)
        self.__comboxes["train.id"].addItems(idl_results_folders)
        self.__comboxes["train.id"].activated.connect(self.__choose_idl_id)
        self._btn_prev_round.clicked.connect(self.__choose_prev_round)
        self._btn_next_round.clicked.connect(self.__choose_next_round)
        for i in ["bright", "contrast"]:
            self.__sliders[i].valueChanged.connect(self.__refresh_imgs)
        for i in ["transverse", "coronal", "sagittal"]:
            self.__radio_btns[i].toggled.connect(self.__set_img_plane)

        # set initial state
        self.__comboxes["patient"].setEnabled(False)
        self.__comboxes["round"].setEnabled(False)
        self._btn_prev_round.setEnabled(False)
        self._btn_prev_round.setArrowType(Qt.LeftArrow)
        self._btn_next_round.setEnabled(False)
        self._btn_next_round.setArrowType(Qt.RightArrow)
        self.__radio_btns["transverse"].setChecked(True)
        self.__radio_btns["coronal"].setChecked(False)
        self.__radio_btns["sagittal"].setChecked(False)

        # set slider range and default value
        self.__sliders["bright"].setMinimum(-128)
        self.__sliders["bright"].setMaximum(128)
        self.__sliders["bright"].setValue(0)
        self.__sliders["contrast"].setMinimum(0)
        self.__sliders["contrast"].setMaximum(200)
        self.__sliders["contrast"].setValue(100)

    def __set_img_plane(self):
        for i in ["transverse", "coronal", "sagittal"]:
            if self.__radio_btns[i].isChecked():
                self.__img_plane = i

                # update and check slice_id (starts from 0)
                img_depth = self.__get_img_depth()
                if img_depth is not None:
                    self.__slice_id = round(img_depth / 2) - 1
                    self.__slice_id = g.check_limit(self.__slice_id, 0, img_depth - 1)

                self.__reset_zoomin()
                self.__refresh_imgs()
                self.__refresh_title()

    def __clear_img_frames(self):
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            width = self.__img_frames[i].width()
            height = self.__img_frames[i].height()
            black_img = np.zeros([width, height, 3])
            qt_image = QImage(
                black_img,
                self.__img_frames[i].width(),
                self.__img_frames[i].height(),
                self.__img_frames[i].width() * 3,
                QImage.Format_RGB888,
            )
            self.__img_frames[i].setPixmap(QPixmap.fromImage(qt_image))

    def __choose_idl_id(self):
        self.__reset_zoomin()
        self.__comboxes["patient"].clear()
        self.__comboxes["round"].setEnabled(False)
        self._btn_prev_round.setEnabled(False)
        self._btn_next_round.setEnabled(False)
        self.__comboxes["round"].clear()
        self.__clear_img_data()
        self.__idl_id = self.__comboxes["train.id"].currentText()
        patient_list = g.get_sub_folders(
            os.path.join(g.IDL_RESULTS_FOLDER, self.__idl_id)
        )
        self.__comboxes["patient"].addItems(patient_list)
        self.__comboxes["patient"].activated.connect(self.__choose_patient)
        self.__comboxes["patient"].setEnabled(True)

        # dont reset patient when train id changed
        self.__choose_patient(reset_patient=False)

    def __choose_patient(self, reset_patient: bool = True):
        self.__reset_zoomin()

        # triggered by idl_id combox changed
        if reset_patient is False:
            # see if cur_patient_id is in patient_combobox
            patient_list = g.get_combox_content(self.__comboxes["patient"])
            if self.__patient not in patient_list:
                self.__patient = self.__comboxes["patient"].currentText()
            else:
                self.__comboxes["patient"].setCurrentText(self.__patient)
        # triggered by patient combox changed
        else:
            self.__patient = self.__comboxes["patient"].currentText()

        cur_patient_folder = os.path.join(
            g.IDL_RESULTS_FOLDER, self.__idl_id, self.__patient
        )

        # load baseline img
        baseline_folder = os.path.join(
            cur_patient_folder,
            "baseline",  # "round=00",
        )
        for i in ["ct", "pt", "mrt1", "mrt2", "label"]:
            cur_img_path = os.path.join(baseline_folder, i + ".nii")
            self.__img_data[i] = self.__load_img(cur_img_path)

        # load round combobox
        self.__comboxes["round"].clear()
        round_list = ["baseline"]
        for i in ["filter.fp", "post.process"]:
            if os.path.exists(os.path.join(cur_patient_folder, i)):
                round_list.append(i)
        round_list += g.get_sub_folders(
            os.path.join(g.IDL_RESULTS_FOLDER, self.__idl_id, self.__patient),
            key_word="round=",
        )
        # for i in round_list:
        self.__comboxes["round"].addItems(round_list)
        self.__comboxes["round"].activated.connect(self.__choose_round)
        self.__comboxes["round"].setEnabled(True)

        # update slice id
        # if this is first time after initialization, slice_is=None,
        # show the middle slice of whole 3d img
        # otherwise try to keep slice_id unchanged
        img_depth = self.__get_img_depth()
        if img_depth is not None:
            if self.__slice_id is None:
                self.__slice_id = round(img_depth / 2) - 1
            # check slice_id (starts from 0)
            self.__slice_id = g.check_limit(self.__slice_id, 0, img_depth - 1)

        # automatically choose round
        self.__choose_round(reset_round=False)

    def __choose_round(self, reset_round: bool = True):
        round_list = g.get_combox_content(self.__comboxes["round"])
        if reset_round is False:
            # see if round is in comboxes["round"]
            if self.__round not in round_list:
                self.__round = self.__comboxes["round"].currentText()
            else:
                self.__comboxes["round"].setCurrentText(self.__round)
        else:
            self.__round = self.__comboxes["round"].currentText()

        # load pred img
        cur_round_folder = os.path.join(
            g.IDL_RESULTS_FOLDER,
            self.__idl_id,
            self.__patient,
            self.__round,
        )
        pred_img_path = os.path.join(cur_round_folder, "pred.nii")
        self.__img_data["pred"] = self.__load_img(pred_img_path)
        self.__img_data["pred"] = g.binarize_img(self.__img_data["pred"])

        # load dsc/msd/hd95
        iter_file_name = g.get_sub_files(
            cur_round_folder, key_word=".json", shuffle=False
        )
        # iter json file exists
        if len(iter_file_name) > 0:
            iter_file_name = iter_file_name[-1]
            json_data = g.load_json(os.path.join(cur_round_folder, iter_file_name))
            for i in ["dsc", "msd", "hd95"]:
                self.__score[i] = json_data[i]["3d"]
        # iter json file not exist
        else:
            for i in ["dsc", "msd", "hd95"]:
                self.__score[i] = None

        # enable/disable prev/next round buttons
        idx = self.__comboxes["round"].currentIndex()
        if idx == 0:
            self._btn_prev_round.setEnabled(False)
            if len(round_list) > 1:
                self._btn_next_round.setEnabled(True)
            else:
                self._btn_next_round.setEnabled(False)

        elif idx == (self.__comboxes["round"].count() - 1):
            self._btn_prev_round.setEnabled(True)
            self._btn_next_round.setEnabled(False)

        else:
            self._btn_prev_round.setEnabled(True)
            self._btn_next_round.setEnabled(True)

        # update ui
        self.__refresh_imgs()
        self.__refresh_title()

    def __choose_prev_round(self):
        idx = self.__comboxes["round"].currentIndex() - 1
        if idx < 0:
            return
        prev_round = self.__comboxes["round"].itemText(idx)
        self.__comboxes["round"].setCurrentText(prev_round)
        self.__choose_round(reset_round=True)

    def __choose_next_round(self):
        idx = self.__comboxes["round"].currentIndex() + 1
        if idx > self.__comboxes["round"].count() - 1:
            return
        next_round = self.__comboxes["round"].itemText(idx)
        self.__comboxes["round"].setCurrentText(next_round)
        self.__choose_round(reset_round=True)

    def __refresh_imgs(self):
        # no img data loaded
        if self.__img_data["ct"] is None:
            return

        # check if cur slice is annotated
        if self.__img_plane == "transverse":
            is_annotated = self.__is_annotated()

        # set contour color
        color = NestedDict()
        color["label"] = self.__color["label"]

        if self.__img_plane == "transverse" and is_annotated:
            color["pred"] = self.__color["annotated"]
        else:
            color["pred"] = self.__color["pred"]

        for i in ["ct", "pt", "mrt1", "mrt2"]:
            # load img
            if self.__img_plane == "sagittal":
                rgb_img = self.__img_data[i][:, :, self.__slice_id]
            elif self.__img_plane == "coronal":
                rgb_img = self.__img_data[i][:, self.__slice_id, :]
            else:  # img_plane == "transverse":
                # for transverse plane, img is upside down,
                # true slice id is: img_depth - 1 - slice_id
                rgb_img = self.__img_data[i][
                    (self.__get_img_depth() - 1 - self.__slice_id), :, :
                ]

            rgb_img = np.uint8((rgb_img - rgb_img.min()) / rgb_img.ptp() * 255.0)
            # after cv2.cvtColor, rgb_img has 3 channels, but is still numpy
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)

            # brightness and contrast
            bright = self.__sliders["bright"].value()
            contrast = self.__sliders["contrast"].value() / 100
            blank = np.zeros_like(rgb_img)
            # cv2.addWeighted: dst = src1 * alpha + src2 * beta + gamma
            rgb_img = cv2.addWeighted(rgb_img, contrast, blank, 0, bright)

            # add mask to annotated slices on "sagittal" or "coronal"
            if self.__img_plane == "sagittal" or self.__img_plane == "coronal":
                total_slices_num = self.__img_data["ct"].shape[0]
                rgb_img_zeros = np.zeros((rgb_img.shape), dtype=np.uint8)

                for cur_annotated_slice in self.__get_annotated_slices():
                    cur_annotated_slice = int(cur_annotated_slice)

                    # mask
                    cur_annotated_slice_mask = cv2.rectangle(
                        rgb_img_zeros,
                        (0, total_slices_num - (cur_annotated_slice - 1)),
                        (
                            rgb_img.shape[1] - 1,
                            total_slices_num - (cur_annotated_slice - 1),
                        ),
                        color=self.__color["annotated"],
                        thickness=-1,
                    )
                    rgb_img = cv2.addWeighted(
                        src1=rgb_img,
                        alpha=1,
                        src2=cur_annotated_slice_mask,
                        beta=1,  # 0.5,
                        gamma=0,
                    )
                    # break
                pass

            # resize and fit img frame
            rgb_img, self.__resize_pos[i] = self.__fit_img_frame(
                rgb_img, self.__img_frames[i]
            )
            # blur after __fit_img_frame will gain better effect
            rgb_img = cv2.GaussianBlur(rgb_img, (3, 3), cv2.BORDER_DEFAULT)

            # draw label and prediction contour
            for k in ["label", "pred"]:
                if self.__img_plane == "sagittal":
                    contour = self.__img_data[k][:, :, self.__slice_id].astype(np.uint8)
                elif self.__img_plane == "coronal":
                    contour = self.__img_data[k][:, self.__slice_id, :].astype(np.uint8)
                else:  # img_plane == "transverse":
                    # for transverse plane, img is upside down,
                    # true slice id is: img_depth - 1 - slice_id
                    contour = self.__img_data[k][
                        (self.__get_img_depth() - 1 - self.__slice_id), :, :
                    ].astype(np.uint8)

                contour, _ = self.__fit_img_frame(contour, self.__img_frames[i])
                # blur after __fit_img_frame will gain better effect
                contour = cv2.GaussianBlur(contour, (7, 7), cv2.BORDER_DEFAULT)
                contour, _ = cv2.findContours(
                    contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                rgb_img = cv2.drawContours(
                    image=rgb_img,
                    contours=contour,
                    contourIdx=-1,
                    color=color[k],
                    thickness=2,
                )

            # add text dsc/msd/hd95
            height = rgb_img.shape[0]
            width = rgb_img.shape[1]
            channel = rgb_img.shape[2]
            cv_text = ""

            if g.is_number(self.__score["dsc"]):
                cv_text += "3D DSC: {:.3f}".format(self.__score["dsc"])
            else:
                cv_text += "3D DSC: N/A"
            cv_text += "\n"

            if g.is_number(self.__score["msd"]):
                cv_text += "3D MSD: {:.2f}".format(self.__score["msd"])
            else:
                cv_text += "3D MSD: N/A"
            cv_text += "\n"

            if g.is_number(self.__score["hd95"]):
                cv_text += "3D HD95: {:.2f}".format(self.__score["hd95"])
            else:
                cv_text += "3D HD95: N/A"

            self.__cv_put_text(
                img=rgb_img,
                text=cv_text,
                pos=(10, 10),
                color=self.__color["score.text"],
            )

            # add text: ground truth/prediction
            cv_text = "GROUND TRUTH"
            self.__cv_put_text(
                img=rgb_img,
                text=cv_text,
                pos=(10, height - 50),
                color=color["label"],
            )
            cv_text = "\n"
            cv_text += "PREDICTION"
            if self.__img_plane == "transverse" and is_annotated:
                cv_text += " (ANNOTATED)"
            self.__cv_put_text(
                img=rgb_img,
                text=cv_text,
                pos=(10, height - 50),
                color=color["pred"],
            )

            # show imgs
            qt_image = QImage(
                rgb_img,
                width,
                height,
                width * channel,
                QImage.Format_RGB888,
            )
            self.__img_frames[i].setPixmap(QPixmap.fromImage(qt_image))

    def __get_annotated_slices(self) -> list:
        # get current round

        if self.__round.startswith("round="):
            cur_round = int(self.__round[len("round=") :])
            if cur_round == 0:
                return []
        else:
            return []

        # load annotated slices
        json_path = os.path.join(
            g.IDL_RESULTS_FOLDER,
            self.__idl_id,
            self.__patient,
            "annotated_slices.json",
        )
        if not os.path.exists(json_path):
            return []

        annotated_slice_dict = g.load_json(json_path)
        if cur_round > len(annotated_slice_dict):
            cur_round = len(annotated_slice_dict)

        annotated_slice_list = []
        for cur_round in annotated_slice_dict:
            annotated_slice_list += g.str_to_list(annotated_slice_dict[cur_round])
            if cur_round == self.__round:
                break

        return annotated_slice_list

    def __is_annotated(self) -> bool:
        if self.__img_data["ct"] is None:
            return False

        # only run in "transverse" plane
        if self.__img_plane == "sagittal" or self.__img_plane == "coronal":
            return False

        annotated_slices = self.__get_annotated_slices()

        # change slice_id from number to str "0xx"
        if "{:03d}".format(self.__slice_id) in annotated_slices:
            return True
        else:
            return False

    def __cv_put_text(
        self,
        img,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
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
                fontScale=1.2,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def wheelEvent(self, event):
        if self.__slice_id is not None:
            img_depth = self.__get_img_depth()
            if img_depth is None:
                return
            self.__slice_id = (
                self.__slice_id - event.angleDelta().y() // 120
            ) % img_depth
            self.__refresh_imgs()
            self.__refresh_title()

    def __get_img_depth(self):
        if (self.__img_data["ct"] is None) or (self.__img_plane is None):
            return None
        if self.__img_plane == "sagittal":
            img_depth = self.__img_data["ct"].shape[2]
        elif self.__img_plane == "coronal":
            img_depth = self.__img_data["ct"].shape[1]
        else:  # img_plane == "transverse"
            img_depth = self.__img_data["ct"].shape[0]
        return img_depth

    def __refresh_title(self):
        win_tital = "iDL.Tool "
        if self.__idl_id is not None:
            win_tital += "   Training.ID=" + self.__idl_id
        if self.__patient is not None:
            win_tital += "   Patient=" + self.__patient[len("patient=") :]
        if self.__round is not None:
            win_tital += "   Num.of.Annotated.Slices="
            win_tital += str(len(self.__get_annotated_slices()))
        if self.__slice_id is not None:
            img_depth = self.__get_img_depth()
            if img_depth is not None:
                win_tital += "   Slice={}/{}".format(self.__slice_id + 1, img_depth)
        self.setWindowTitle(win_tital)

    def resizeEvent(self, event):
        side_bar_width = 300
        self.__refresh_img_frames_size(side_bar_width)
        self.__refresh_side_bar(side_bar_width)
        self.__refresh_imgs()
        QMainWindow.resizeEvent(self, event)

    def __refresh_side_bar(self, side_bar_width: int):
        # these are adjustable
        left = 30
        top = 0
        text_label_height = 25
        gap = 80
        combox_height = 30
        radio_btn_height = 25
        # side bar location
        side_bar_x = self.geometry().width() - side_bar_width
        width = side_bar_width - left * 2
        left += side_bar_x

        # idl_id / patient
        for i in ["train.id", "patient"]:
            top += gap
            rect = QRect(left, top, width, text_label_height)
            self.__text_labels[i].setGeometry(rect)
            top += text_label_height
            rect = QRect(left, top, width, combox_height)
            self.__comboxes[i].setGeometry(rect)

        # text label of "round"
        top += gap
        rect = QRect(left, top, width, text_label_height)
        self.__text_labels["round"].setGeometry(rect)

        # btn and text_box of "round"
        top += text_label_height
        btn_width = 30
        annotat_ui_left = left
        rect = QRect(annotat_ui_left, top - 1, btn_width, combox_height + 2)
        self._btn_prev_round.setGeometry(rect)
        annotat_ui_left += btn_width
        rect = QRect(annotat_ui_left, top, width - btn_width * 2, combox_height)
        self.__comboxes["round"].setGeometry(rect)
        annotat_ui_left += width - btn_width * 2
        rect = QRect(annotat_ui_left, top - 1, btn_width, combox_height + 2)
        self._btn_next_round.setGeometry(rect)

        # brightness and contrast
        for i in ["bright", "contrast"]:
            top += gap
            rect = QRect(left, top, width, text_label_height)
            self.__text_labels[i].setGeometry(rect)
            top += text_label_height
            rect = QRect(left, top, width, combox_height)
            self.__sliders[i].setGeometry(rect)

        # img plane
        top += gap
        for i in ["transverse", "coronal", "sagittal"]:
            rect = QRect(left, top, width, radio_btn_height)
            self.__radio_btns[i].setGeometry(rect)
            top += radio_btn_height

    def __refresh_img_frames_size(self, side_bar_width: int):
        gap = 1
        size = NestedDict()
        size["x"] = self.geometry().width() - side_bar_width
        size["y"] = self.geometry().height() - self._menu_bar.height()
        for i in ["x", "y"]:
            double_size = size[i] - gap * 3
            size.pop(i)
            size[i][0] = double_size // 2
            size[i][1] = double_size // 2
            if double_size % 2 != 0:
                size[i][0] += 1

        pos = NestedDict()
        for i in ["x", "y"]:
            pos[i][0] = gap
            pos[i][1] = size[i][0] + gap * 2

        self.__img_frames["ct"].setGeometry(
            QRect(pos["x"][0], pos["y"][0], size["x"][0], size["y"][0])
        )
        self.__img_frames["pt"].setGeometry(
            QRect(pos["x"][1], pos["y"][0], size["x"][1], size["y"][0])
        )
        self.__img_frames["mrt1"].setGeometry(
            QRect(pos["x"][0], pos["y"][1], size["x"][0], size["y"][1])
        )
        self.__img_frames["mrt2"].setGeometry(
            QRect(pos["x"][1], pos["y"][1], size["x"][1], size["y"][1])
        )

    def __open_file_dlg(self):
        Tk().withdraw()
        file_name = filedialog.askopenfilename()
        if file_name == "" or file_name is None:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
