import enum
import os

import cv2
import numpy as np
from custom import Dict, Directory
from custom import Global as g
from custom import Img, Json, List, Nii, Time, Value
from PyQt5.QtCore import QRect, Qt
from ui_replay import UiReplay


class IdlStep(enum.Enum):
    CHOOSE_PATIENT = 0
    CLICK_GTVT_CENTER = 1
    ANNOTATE_GTVT = 2
    CLICK_GTVN_CENTER = 3
    IDL_COMPLETE = 4


class UiIdl(UiReplay):
    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_F12:
            pass

    def _init_member_var(self):
        super()._init_member_var()
        self.__idl_step = IdlStep.CHOOSE_PATIENT
        # keep idl.gtvt/gtvn id unchanged
        cur_time = Time.get_cur_time_str()
        self._idl_gtvt_id = "idl.gtvt_" + cur_time
        self._idl_gtvn_id = "idl.gtvn_" + cur_time

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def __confirm_annotation(self):
        if self.__annotation_step == IdlStep.CLICK_GTVT_CENTER:
            self._text_box_annotation_msg.setText(
                "Please delineate the GTVt on transvers/coronal/sagittal plane, then press the 'Confirm' button (green button)"
            )
            self.__annotation_step = IdlStep.ANNOTATE_GTVT

        elif self.__annotation_step == IdlStep.ANNOTATE_GTVT:
            self._text_box_annotation_msg.setText(
                "Please click the center of GTVns, then press the 'Confirm' button (green button)"
            )
            # gtvt training process
            ###################################
            # self._combox["idl.gtvt"].addItem(new_gtvt_id)
            # self._combox["idl.gtvt"].setCurrentText(new_gtvt_id)
            self.__annotation_step = IdlStep.CLICK_GTVN_CENTER

        elif self.__annotation_step == IdlStep.CLICK_GTVN_CENTER:
            # gtvn training process
            ###################################
            # self._clear_img_data()
            # reset ui
            # for i in ["idl.gtvs", "idl.gtvt", "idl.gtvn", "patient", "round"]:
            #     self._combox[i].clear()
            #     self._combox[i].setEnabled(False)
            #     self._arrow_btn["prev.{}".format(i)].setEnabled(False)
            #     self._arrow_btn["next.{}".format(i)].setEnabled(False)

            self._combox["idl.gtvn"].setCurrentText("idl.gtvn_2023.05.26.12.06.15_best")
            self._choose_patient(idx=None, reset_patient=False)
            self.__annotation_step = IdlStep.IDL_COMPLETE
        else:
            pass

    def __clear_annotation(self):
        print("clear annotation")

    def __select_pen(self):
        print("select pen")

    def __select_eraser(self):
        print("select eraser")

    def _init_ui_names(self):
        super()._init_ui_names()
        self._text_label["annotation.tools"] = self._text_label_annotation_tools
        self._text_label["idl.gtvt.progress"] = self._text_label_idl_gtvt_progress

        self.__btn = Dict()
        self.__btn["pen"] = self._btn_pen
        self.__btn["eraser"] = self._btn_eraser
        self.__btn["clear"] = self._btn_clear
        self.__btn["confirm"] = self._btn_confirm

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

        # idl gtvt retraining progress
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label_idl_gtvt_progress.setGeometry(rect)
        self._text_label_idl_gtvt_progress.show()
        top += text_height
        rect = QRect(left, top, width, bar_height)
        self._progress_bar_idl_gtvt.setGeometry(rect)
        self._progress_bar_idl_gtvt.show()

    def _choose_baseline(self):
        self._reset_zoomin()
        self._clear_img_data()
        self._clear_display_frames()

        self._baseline_id = "baseline_real.idl"
        # run these 2 lines after self._baseline_id is confirmed
        self._load_dataset_dir_and_nii_spacing()
        self._fill_combox_patient()

        # run this after patient combox current text is set up
        self._enable_arrow_btns("patient")

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

        cv_text = "GTVt"
        pos_y += gap_y
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["gtvt.pred"],
        )

        cv_text = "GTVn"
        pos_y += gap_y
        self._cv_put_text(
            img=rgb_img,
            text=cv_text,
            pos=(pos_x, pos_y),
            color=self._color["gtvn.pred"],
        )

    def _choose_patient(self, idx: int = None):
        self._patient = self._combox["patient"].currentText()
        self._reset_zoomin()

        # load multi-modal imgs only, no need to load labels
        self._load_multi_modal_imgs()

        # get slice id (after multi-modal imgs are loaded)
        self._cur_slice = self._get_middle_slice_id()

        self._choose_idl_gtvt()
        self._choose_idl_gtvn()
        self._refresh_imgs()
        self._refresh_title()

        self._text_box_annotation_msg.setText(
            "Click the center of the GTVt, then press the 'Confirm' button (green button)"
        )
        # update annotation step
        self.__annotation_step = IdlStep.CLICK_GTVT_CENTER

    def _choose_idl_gtvt(self):
        idl_gtvt_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_gtvt_id
        )
        # find idl.gtvt result
        if os.path.exists(idl_gtvt_dir):
            # choose the last round
            gtvt_pred_path = Directory.get_sub_folders(
                os.path.join(
                    idl_gtvt_dir,
                    "patients",
                    "patient={}".format(self._patient),
                ),
                key_word="round=",
                full_path=True,
            )[-1]
        # cant find idl.gtvt result, then load baseline pred
        else:
            gtvt_pred_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "patients",
                "patient={}".format(self._patient),
            )
        gtvt_pred_path = os.path.join(gtvt_pred_path, "gtvt_pred.nii")
        self._3d_imgs["gtvt.pred"] = self._load_img(gtvt_pred_path)
        self._3d_imgs["gtvt.pred"] = Img.binarize(self._3d_imgs["gtvt.pred"])

    def _choose_idl_gtvn(self):
        idl_gtvn_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_gtvn_id
        )
        # find idl.gtvt result
        if os.path.exists(idl_gtvn_dir):
            # choose the last round
            gtvn_pred_path = Directory.get_sub_folders(
                os.path.join(
                    idl_gtvn_dir,
                    "patients",
                    "patient={}".format(self._patient),
                ),
                key_word="round=",
                full_path=True,
            )[-1]
        # cant find idl.gtvt result, then load baseline pred
        else:
            gtvn_pred_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                "baseline",
                "patients",
                "patient={}".format(self._patient),
            )
        gtvn_pred_path = os.path.join(gtvn_pred_path, "gtvn_pred.nii")
        self._3d_imgs["gtvn.pred"] = self._load_img(gtvn_pred_path)
        self._3d_imgs["gtvn.pred"] = Img.binarize(self._3d_imgs["gtvn.pred"])
