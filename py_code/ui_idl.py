import os
import random

import cv2
import numpy as np
import qimage2ndarray
from custom import Debug, Dict, Dir, DrawingMode
from custom import Global as g
from custom import IDLStep, Img, Json, List, Nii, Time, Value
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtGui import (
    QColor,
    QCursor,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import QButtonGroup, QMessageBox, QRadioButton
from scipy import ndimage
from str_lib import CORONAL, CT, MR1, MR2, PT, SAGITTAL, TRANSVERSE
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt
from ui_draggable_cross import DraggableCross
from ui_replay import UiReplay


class UiIDL(UiReplay):
    def draw_on_4_qlabels_press(self, event: QMouseEvent):
        idl_step = self.get_cur_patient_idl_step()
        if idl_step not in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
            return

        if idl_step == IDLStep.DRAW_GTVT:
            center_slice_id = self.__get_gtvt_center_slice_id()

            # if on center slice, start painting
            if self.cur_slice_id == center_slice_id:
                self.paint_pos = event.pos()

            # if on other slices, switch to center slice
            else:
                self.cur_slice_id = center_slice_id
                self.refresh_img_qlabels()

        elif idl_step == IDLStep.CORRECTION:
            self.paint_pos = event.pos()

    def draw_on_4_qlabels_move(self, event: QMouseEvent):
        if self.paint_pos is None:
            return

        pen_size = self.get_pen_size()
        eraser_size = pen_size + 2
        eraser_color = QColor(*self._color["eraser"])
        for i in [CT, PT, MR1, MR2]:
            painter = QPainter(self.img_qlabel[i].drawing_layer)

            # if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            #     painter.setCompositionMode(QPainter.CompositionMode_Clear)
            #     painter.setPen(
            #         QPen(Qt.transparent, pen_size + 2, Qt.SolidLine, Qt.RoundCap)
            #     )

            # smooth
            painter.setRenderHint(QPainter.Antialiasing)
            # Set the composition mode to control alpha blending
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                pen_color = QColor(*self._color["gtvt.pred"])
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                pen_color = QColor(*self._color["gtvn.pred"])

            if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
                painter.setPen(
                    QPen(eraser_color, eraser_size, Qt.SolidLine, Qt.RoundCap)
                )
                self.img_qlabel[i].pen_mode = False
            elif self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
                painter.setPen(QPen(pen_color, pen_size, Qt.SolidLine, Qt.RoundCap))
                self.img_qlabel[i].pen_mode = True

            painter.drawLine(self.paint_pos, event.pos())

            self.img_qlabel[i].update()  # schedule a repaint

        self.paint_pos = event.pos()  # update paint pos

    def draw_on_4_qlabels_release(self):
        if self.paint_pos is None:
            return

        # binarize threshold
        # this is for saving qimage as ndarray
        # binarization is needed before and after resize the ndarray
        binary_threshold = 0.5

        # save drawing layer into 2d ndarray
        # qpixmap to a qimage
        qimg = self.img_qlabel[CT].drawing_layer.toImage()
        # qimage to ndarray
        annotation_2d = qimage2ndarray.alpha_view(qimg).astype(np.float32)
        annotation_2d /= 255

        # binarization (before resize)
        annotation_2d = Img.binarize(img=annotation_2d, threshold=binary_threshold)

        # crop annotation_2d based on roi
        x = self._rgb_img_roi["x"]
        y = self._rgb_img_roi["y"]
        width = self._rgb_img_roi["width"]
        height = self._rgb_img_roi["height"]
        annotation_2d = annotation_2d[y : y + height, x : x + width]

        # resize to actual size
        if self._plane == SAGITTAL:
            actual_shape = self.img_3d[CT][:, :, 0].shape
        elif self._plane == CORONAL:
            actual_shape = self.img_3d[CT][:, 0, :].shape
        elif self._plane == TRANSVERSE:
            actual_shape = self.img_3d[CT][0, :, :].shape
        annotation_2d = cv2.resize(
            annotation_2d,
            (actual_shape[1], actual_shape[0]),
            interpolation=cv2.INTER_AREA,  # best for scaling down
        )

        # binarization (after resize)
        annotation_2d = Img.binarize(img=annotation_2d, threshold=binary_threshold)

        # add annotation_2d on 3d annotation
        idl_step = self.get_cur_patient_idl_step()
        if idl_step == IDLStep.DRAW_GTVT:
            t, c, s = self.__gtvt_click_pos_3d
            if self._plane == TRANSVERSE:
                segment = self.img_3d["gtvt.annotation"][t, :, :]
            elif self._plane == CORONAL:
                segment = self.img_3d["gtvt.annotation"][:, c, :]
            elif self._plane == SAGITTAL:
                segment = self.img_3d["gtvt.annotation"][:, :, s]

        elif idl_step == IDLStep.CORRECTION:
            t = c = s = self.cur_slice_id
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
                # segment_type_list = ["correction", "annotation", "pred"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
                # segment_type_list = ["correction", "pred"]

            # # loop through correction->annotation->pred until finding un-empty slice
            # for i in segment_type_list:
            #     _3d_img = self.img_3d["{}.{}".format(gtv, i)]
            #     if self._plane == TRANSVERSE:
            #         segment = _3d_img[t, :, :].copy()
            #     elif self._plane == CORONAL:
            #         segment = _3d_img[:, c, :].copy()
            #     elif self._plane == SAGITTAL:
            #         segment = _3d_img[:, :, s].copy()

            #     if i != "pred":
            #         kernel = np.ones((3, 3), np.uint8)
            #         eroded_segment = cv2.erode(segment, kernel, iterations=1)
            #         if eroded_segment.max() <= 0:
            #             continue
            #         else:
            #             break
            #     else:
            #         if segment.max() <= 0:
            #             continue
            #         else:
            #             break

            _3d_img = self.img_3d["{}.pred.final".format(gtv)]
            if self._plane == TRANSVERSE:
                segment = _3d_img[t, :, :].copy()
            elif self._plane == CORONAL:
                segment = _3d_img[:, c, :].copy()
            elif self._plane == SAGITTAL:
                segment = _3d_img[:, :, s].copy()

        # invert color if in eraser mode
        if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            segment = 1 - segment

        # combine annotation_2d and segment
        segment = np.maximum(segment, annotation_2d)

        # fill holes if in pen mode
        if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
            segment = ndimage.binary_fill_holes(segment).astype(np.float32)

        # invert color back, if in eraser mode
        if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            segment = 1 - segment

        # replace slice in 3d gtvt.annotation or gtvt/gtvn correction
        if idl_step == IDLStep.DRAW_GTVT:
            _3d_img = self.img_3d["gtvt.annotation"]
        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                _3d_img = self.img_3d["gtvt.correction"]
                _3d_mask = self.img_3d["gtvt.correction.mask"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                _3d_img = self.img_3d["gtvn.correction"]
                _3d_mask = self.img_3d["gtvn.correction.mask"]

        # replace slice
        if self._plane == TRANSVERSE:
            _3d_img[t, :, :] = segment
        elif self._plane == CORONAL:
            _3d_img[:, c, :] = segment
        elif self._plane == SAGITTAL:
            _3d_img[:, :, s] = segment

        # update correction mask
        if idl_step == IDLStep.CORRECTION:
            if self._plane == TRANSVERSE:
                if segment.max() == 0:
                    _3d_mask[t, :, :] = np.zeros_like(segment)
                else:
                    _3d_mask[t, :, :] = np.ones_like(segment)
            elif self._plane == CORONAL:
                if segment.max() == 0:
                    _3d_mask[:, c, :] = np.zeros_like(segment)
                else:
                    _3d_mask[:, c, :] = np.ones_like(segment)
            elif self._plane == SAGITTAL:
                if segment.max() == 0:
                    _3d_mask[:, :, s] = np.zeros_like(segment)
                else:
                    _3d_mask[:, :, s] = np.ones_like(segment)

        # save gtvt/gtvn corrections and correction masks
        if idl_step == IDLStep.CORRECTION:
            for gtv in ["gtvt", "gtvn"]:
                self.__save_corrections(gtv)

        # update values
        self.paint_pos = None
        self.__update_gtvt_annotated_status()
        self.__combine_pred_annotation_correction()

        # update UI
        self.__clear_all_drawing_layers_on_4_qlabels()
        self.refresh_img_qlabels()

    def __save_corrections(self, gtv: str):
        if gtv not in ["gtvt", "gtvn"]:
            Debug.error_exit("gtv value error")

        if self.img_3d["{}.correction".format(gtv)] is None:
            return

        cur_patient_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id[gtv],
            "patients",
            "patient={}".format(self._cur_patient),
        )
        cur_round_dir = os.path.join(
            cur_patient_dir,
            "round=01",
        )

        for i in ["correction", "correction.mask"]:
            img = self.img_3d["{}.{}".format(gtv, i)].copy()
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                img = np.flip(img, axis=2)
            # turn upside down
            img = np.flip(img, axis=0)
            # save
            Nii.save(
                img=img,
                save_path=os.path.join(
                    cur_round_dir, "{}_{}.nii.gz".format(gtv, i.replace("_", "."))
                ),
                spacing=self._nii_spacing,
            )

    def __click_btn_pen(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.DRAW_GTVT:
            self.drawing_mode = DrawingMode.GTVT_PEN

        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode == DrawingMode.GTVT_ERASER:
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self.drawing_mode == DrawingMode.GTVN_ERASER:
                self.drawing_mode = DrawingMode.GTVN_PEN

        if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
            self.__set_mouse_cursor("pen")
            self._text_label["pen.size"].setText("Pen Size")

    def __click_btn_eraser(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.DRAW_GTVT:
            self.drawing_mode = DrawingMode.GTVT_ERASER

        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode == DrawingMode.GTVT_PEN:
                self.drawing_mode = DrawingMode.GTVT_ERASER
            elif self.drawing_mode == DrawingMode.GTVN_PEN:
                self.drawing_mode = DrawingMode.GTVN_ERASER

        if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
            self.__set_mouse_cursor("eraser")
            self._text_label["pen.size"].setText("Eraser Size")

    def __click_btn_confirm(self):
        if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER:
            if self.__gtvt_click_pos_3d is None:
                QMessageBox.information(
                    self,
                    "Information",
                    "GTVt center not detected.",
                    QMessageBox.Ok,
                )
                return

            # add clicks into 3d img
            pos = self.__gtvt_click_pos_3d
            if self.img_3d["gtvt.click"] is None:
                self.img_3d["gtvt.click"] = np.zeros_like(self.img_3d[CT])
            # pos 0-transverse 1-coronal 2-saggital
            self.img_3d["gtvt.click"][pos[0]][pos[1]][pos[2]] = 1

            # save gtvt_click
            cur_patient_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id["gtvt"],
                "patients",
                "patient={}".format(self._cur_patient),
            )
            cur_round_dir = os.path.join(
                cur_patient_dir,
                "round=01",
            )
            Dir.create(cur_round_dir)
            idl_gtvt_click = self.img_3d["gtvt.click"].copy()
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                idl_gtvt_click = np.flip(idl_gtvt_click, axis=2)
            # turn upside down
            idl_gtvt_click = np.flip(idl_gtvt_click, axis=0)
            Nii.save(
                img=idl_gtvt_click,
                save_path=os.path.join(cur_round_dir, "gtvt_click.nii.gz"),
                spacing=self._nii_spacing,
            )

            # save gtvt selected_slices.json
            pos = np.where(idl_gtvt_click == 1)
            selected_slices = Dict()
            selected_slices[TRANSVERSE]["round=01"] = List(pos[0]).to_str()
            selected_slices[CORONAL]["round=01"] = List(pos[1]).to_str()
            selected_slices[SAGITTAL]["round=01"] = List(pos[2]).to_str()
            Json.save(
                data=selected_slices,
                path=os.path.join(cur_patient_dir, "selected_slices.json"),
            )

            # clean current step elements
            self.delete_all_crosses_on_4_qlabels()
            # new step
            self.set_cur_patient_idl_step(IDLStep.DRAW_GTVT)
            self.__save_idl_step()
            self.refresh_img_qlabels()
            self.img_3d["gtvt.annotation"] = np.zeros_like(self.img_3d[CT])
            self.drawing_mode = DrawingMode.GTVT_PEN
            self.__set_mouse_cursor("pen")
            for i in ["pen", "eraser"]:
                self.__btn[i].setEnabled(True)

        elif self.get_cur_patient_idl_step() == IDLStep.DRAW_GTVT:
            for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
                if self.__gtvt_annotated_status[plane] is False:
                    QMessageBox.information(
                        self,
                        "Information",
                        "Please draw GTVt in {} plane.".format(plane),
                        QMessageBox.Ok,
                    )
                    self._modal_fixed_mode_switch_plane(new_plane=plane)
                    if self.drawing_mode == DrawingMode.GTVT_ERASER:
                        self.drawing_mode = DrawingMode.GTVT_PEN
                        self.__set_mouse_cursor("pen")
                        self._text_label["pen.size"].setText("Pen Size")
                    return

            # save gtvt annotation
            cur_round_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id["gtvt"],
                "patients",
                "patient={}".format(self._cur_patient),
                "round=01",
            )
            Dir.create(cur_round_dir)
            gtvt_annotation_to_save = self.img_3d["gtvt.annotation"].copy()
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                gtvt_annotation_to_save = np.flip(gtvt_annotation_to_save, axis=2)
            # turn upside down
            gtvt_annotation_to_save = np.flip(gtvt_annotation_to_save, axis=0)
            Nii.save(
                img=gtvt_annotation_to_save,
                save_path=os.path.join(cur_round_dir, "gtvt_annotation.nii.gz"),
                spacing=self._nii_spacing,
            )

            # start real idl gtvt
            training_idl_gtvt = TrainingIDLGTVt()
            training_idl_gtvt.new_training(
                baseline_id="baseline_real.idl",
                real_idl_gtvt_id=self._idl_id["gtvt"],
                real_idl_patient=self._cur_patient,
                dataset_ver=self._dataset_ver,
                debug_mode=self.__debug_mode,
            )

            # clean current step elements
            self.__clear_all_drawing_layers_on_4_qlabels()
            # new step
            self.set_cur_patient_idl_step(IDLStep.CLICK_GTVN_CENTER)
            self.__save_idl_step()
            self._load_idl_gtvt_data()
            self.__combine_pred_annotation_correction()
            self.refresh_img_qlabels()
            self.setCursor(Qt.ArrowCursor)
            for i in ["pen", "eraser"]:
                self.__btn[i].setEnabled(False)

        elif self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
            # add clicks into 3d img
            if self.img_3d["gtvn.clicks"] is None:
                self.img_3d["gtvn.clicks"] = np.zeros_like(self.img_3d[CT])
            for pos in self.__gtvn_clicks_pos_3d:
                # pos 0-transverse 1-coronal 2-saggital
                self.img_3d["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1

            # clean current step elements
            self.delete_all_crosses_on_4_qlabels()

            # show gtvn center
            self.refresh_img_qlabels()

            # copy data (dont change origin ndarray)
            idl_gtvn_clicks = self.img_3d["gtvn.clicks"].copy()
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=2)
            # turn upside down
            idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=0)

            # start real idl gtvn
            training_idl_gtvn = TrainingIDLGTVn()
            training_idl_gtvn.real_idl(
                idl_gtvn_id=self._idl_id["gtvn"],
                patient=self._cur_patient,
                idl_gtvn_clicks=idl_gtvn_clicks,
                dataset_part=self._dataset_part,
                dataset_ver=self._dataset_ver,
            )

            # new step
            self.set_cur_patient_idl_step(IDLStep.CORRECTION)
            self.__save_idl_step()
            self._load_idl_gtvn_data()
            self.__combine_pred_annotation_correction()
            self.refresh_img_qlabels()
            # init correction and mask
            for i in [
                "gtvt.correction",
                "gtvn.correction",
                "gtvt.correction.mask",
                "gtvn.correction.mask",
            ]:
                self.img_3d[i] = np.zeros_like(self.img_3d[CT])

            self.drawing_mode = DrawingMode.GTVT_PEN
            self.__set_mouse_cursor("pen")
            for i in ["gtvt", "gtvn"]:
                self._radio_btn["draw.{}".format(i)].setEnabled(True)
            for i in ["pen", "eraser"]:
                self.__btn[i].setEnabled(True)
            self.__btn["confirm"].setEnabled(False)

    # check annotation in 3 different planes
    def __update_gtvt_annotated_status(self) -> Dict:
        t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
        t, c, s = int(t), int(c), int(s)
        for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
            if plane == TRANSVERSE:
                cur_plane_annotation = self.img_3d["gtvt.annotation"][t, :, :].copy()
                cur_plane_annotation[c, :] = 0
                cur_plane_annotation[:, s] = 0

            elif plane == CORONAL:
                cur_plane_annotation = self.img_3d["gtvt.annotation"][:, c, :].copy()
                cur_plane_annotation[t, :] = 0
                cur_plane_annotation[:, s] = 0

            elif plane == SAGITTAL:
                cur_plane_annotation = self.img_3d["gtvt.annotation"][:, :, s].copy()
                cur_plane_annotation[t, :] = 0
                cur_plane_annotation[:, c] = 0

            if cur_plane_annotation.max() == 0:
                self.__gtvt_annotated_status[plane] = False
            else:
                self.__gtvt_annotated_status[plane] = True

    def refresh_img_qlabels(self):
        # no patient loaded
        if self.img_3d[CT] is None:
            # ask user to select a patient
            w = self.img_qlabel[CT].width()
            h = self.img_qlabel[CT].height()
            qimg = QImage(w, h, QImage.Format_RGB888)
            black = QColor(0, 0, 0)
            qimg.fill(black)
            self._add_msg_on_qimg(qimg)
            self.img_qlabel[CT].set_background(qimg)
            self.img_qlabel[CT].update()
            return

        super().refresh_img_qlabels(replay_mode=False)

    def __set_mouse_cursor(self, cursor_type: str):
        if cursor_type not in ["pen", "eraser"]:
            Debug.error_exit("cursor type error")

        cursor_size = 32  # no larger than 32
        cursor_pixmap = QPixmap(
            (os.path.join(g.PROJ_DIR, "icons", "{}_cursor.png".format(cursor_type)))
        )
        cursor_pixmap = cursor_pixmap.scaled(
            cursor_size, cursor_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if cursor_type == "pen":
            self.setCursor(QCursor(cursor_pixmap, 0, cursor_size * 0.95))
        elif cursor_type == "eraser":
            self.setCursor(QCursor(cursor_pixmap, cursor_size * 0.2, cursor_size * 0.8))

    def _init_color(self):
        super()._init_color()
        self._color["gtvt.annotation"] = self._color["yellow"]
        self._color["gtvt.correction"] = self._color["yellow"]
        self._color["gtvn.correction"] = self._color["cyan"]
        self._color["eraser"] = self._color["black"]
        self._color["gtvt.pred.final"] = self._color["gtvt.pred"]
        self._color["gtvn.pred.final"] = self._color["gtvn.pred"]

    def __click_btn_clear(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.clear_gtvt_click_pos_3d()
            self.__refresh_crosses_on_4_qlabels()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.clear_gtvn_clicks_pos_3d()
            self.__refresh_crosses_on_4_qlabels()

        elif idl_step == IDLStep.DRAW_GTVT:
            # clear annotation on cur plane
            t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
            t, c, s = int(t), int(c), int(s)
            # use mask to filter out the annotation on current anatomical plane
            if self._plane == TRANSVERSE:
                mask = np.zeros_like(self.img_3d["gtvt.annotation"][t, :, :])
                mask[c, :] = 1
                mask[:, s] = 1
                self.img_3d["gtvt.annotation"][t, :, :] *= mask
            elif self._plane == CORONAL:
                mask = np.zeros_like(self.img_3d["gtvt.annotation"][:, c, :])
                mask[t, :] = 1
                mask[:, s] = 1
                self.img_3d["gtvt.annotation"][:, c, :] *= mask
            elif self._plane == SAGITTAL:
                mask = np.zeros_like(self.img_3d["gtvt.annotation"][:, :, s])
                mask[t, :] = 1
                mask[:, c] = 1
                self.img_3d["gtvt.annotation"][:, :, s] *= mask

            # update gtvt annotated status
            self.__gtvt_annotated_status[self._plane] = False
            self.__combine_pred_annotation_correction()
            self.refresh_img_qlabels()

        elif idl_step == IDLStep.CORRECTION:
            t = c = s = self.cur_slice_id
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
            _3d_img = self.img_3d["{}.correction".format(gtv)]
            _3d_mask = self.img_3d["{}.correction.mask".format(gtv)]
            if self._plane == TRANSVERSE:
                _3d_img[t, :, :] = np.zeros_like(_3d_img[t, :, :])
                _3d_mask[t, :, :] = np.zeros_like(_3d_mask[t, :, :])
            elif self._plane == CORONAL:
                _3d_img[:, c, :] = np.zeros_like(_3d_img[:, c, :])
                _3d_mask[:, c, :] = np.zeros_like(_3d_mask[:, c, :])
            elif self._plane == SAGITTAL:
                _3d_img[:, :, s] = np.zeros_like(_3d_img[:, :, s])
                _3d_mask[:, :, s] = np.zeros_like(_3d_mask[:, :, s])
            self.__save_corrections(gtv)
            self.__combine_pred_annotation_correction()
            self.refresh_img_qlabels()

    def __get_gtvt_center_slice_id(self):
        if self.__gtvt_click_pos_3d is None:
            Debug.error_exit("no gtvt click")
        if self._plane == TRANSVERSE:
            center_slice_id = self.__gtvt_click_pos_3d[0]
        elif self._plane == CORONAL:
            center_slice_id = self.__gtvt_click_pos_3d[1]
        elif self._plane == SAGITTAL:
            center_slice_id = self.__gtvt_click_pos_3d[2]
        return center_slice_id

    def __get_gtvn_center_slice_id(self):
        if len(self.__gtvn_clicks_pos_3d) == 0:
            Debug.error_exit("no gtvn clicks")
        if self._plane == TRANSVERSE:
            center_slice_id = self.__gtvn_clicks_pos_3d[-1][0]
        elif self._plane == CORONAL:
            center_slice_id = self.__gtvn_clicks_pos_3d[-1][1]
        elif self._plane == SAGITTAL:
            center_slice_id = self.__gtvn_clicks_pos_3d[-1][2]
        return center_slice_id

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__refresh_crosses_on_4_qlabels()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.__refresh_crosses_on_4_qlabels()

    def _modal_fixed_mode_switch_plane(
        self, connected_radio_btn: QRadioButton = None, new_plane: str = None
    ):
        super()._modal_fixed_mode_switch_plane(
            connected_radio_btn=connected_radio_btn, new_plane=new_plane
        )
        self.__refresh_crosses_on_4_qlabels()

    def delete_all_crosses_on_4_qlabels(self):
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].delete_all_crosses()

    def __refresh_crosses_on_4_qlabels(self):
        if (
            self.get_cur_patient_idl_step() != IDLStep.CLICK_GTVT_CENTER
            and self.get_cur_patient_idl_step() != IDLStep.CLICK_GTVN_CENTER
        ):
            return

        # remove old crosses
        self.delete_all_crosses_on_4_qlabels()

        # load crosses position from gtvt/gtvn_clicks_pos_3d
        if (
            self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER
            or self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER
        ):
            if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER:
                if self.__gtvt_click_pos_3d is None:
                    return
                else:
                    clicks_pos_3d = [self.__gtvt_click_pos_3d]
            elif self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
                if len(self.__gtvn_clicks_pos_3d) <= 0:
                    return
                else:
                    clicks_pos_3d = self.__gtvn_clicks_pos_3d

            img_shape = self.get_3d_img_shape()

            for d, h, w in clicks_pos_3d:
                x = y = None
                if self._plane == TRANSVERSE:
                    if self.cur_slice_id == d:
                        x = w / img_shape[2]
                        y = h / img_shape[1]

                elif self._plane == CORONAL:
                    if self.cur_slice_id == h:
                        x = w / img_shape[2]
                        y = d / img_shape[0]

                elif self._plane == SAGITTAL:
                    if self.cur_slice_id == w:
                        x = h / img_shape[1]
                        y = d / img_shape[0]

                # find click on current slice
                if x is not None and y is not None:
                    x *= self._rgb_img_roi["width"]
                    y *= self._rgb_img_roi["height"]
                    x = round(x)
                    y = round(y)
                    x += self._rgb_img_roi["x"]
                    y += self._rgb_img_roi["y"]

                    # do not record click pos when refreshing
                    self.add_4_crosses(QPoint(x, y), record_click_pos=False)

    def delete_click_pos(self, cross: DraggableCross):
        if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER:
            self.__gtvt_click_pos_3d = None
        elif self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
            pos = cross.get_pos_in_3d()
            self.__gtvn_clicks_pos_3d.remove(pos)

    def add_click_pos(self, cross: DraggableCross):
        pos = cross.get_pos_in_3d()
        if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER:
            self.__gtvt_click_pos_3d = pos
        elif self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
            self.__gtvn_clicks_pos_3d.append(pos)

    def set_4_crosses_dragging_offset(self, pos: QPoint):
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].selected_cross.offset = pos

    def set_4_crosses_dragging_state(self, dragging: bool):
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].selected_cross.dragging = dragging

    def move_4_crosses(self, pos: QPoint):
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].selected_cross.move(pos)

    def delete_4_crosses(self):
        cross = self.img_qlabel[CT].selected_cross
        self.delete_click_pos(cross)
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].delete_selected_cross()

    # make this function public, CustomQLabel will use it
    def add_4_crosses(self, pos: QPoint, record_click_pos: bool):
        if self.img_3d[CT] is None:
            return

        # make sure new cross id is unique
        crosses_id_list = self.img_qlabel[CT].get_crosses_id_list()
        while 1:
            cross_id = random.randint(0, 2**16)
            if cross_id not in crosses_id_list:
                break
        # add crosses
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].add_cross(pos=pos, cross_id=cross_id)

        # add clicks into 3d img
        if record_click_pos:
            new_cross = self.img_qlabel[CT].get_cross_by_id(cross_id)
            self.add_click_pos(new_cross)

    def select_4_crosses(self, cross_id: int):
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].select_cross(cross_id)

    def get_rgb_img_roi(self):
        return self._rgb_img_roi

    def get_nii_spacing(self):
        return self._nii_spacing

    def get_img_plane(self):
        return self._plane

    def get_cur_slice(self):
        return self.cur_slice_id

    def get_3d_img_shape(self):
        if self.img_3d[CT] is not None:
            return self.img_3d[CT].shape
        else:
            return None

    def __init__(
        self,
        idl_remark: str = None,
        debug_mode: bool = False,
    ):
        # pass debug_mode parameter to the parent class
        super().__init__(idl_remark=idl_remark, debug_mode=debug_mode)

    def _init_widgets(self):
        super()._init_widgets()

        for i in ["annotation.tools", "idl.progress", "pen.size"]:
            self._text_label[i] = QtWidgets.QLabel(self._central_widget)

        for i in ["gtvt", "gtvn"]:
            self._radio_btn["draw.{}".format(i)] = QRadioButton()

        self.__btn = Dict()
        for i in ["pen", "eraser", "clear", "confirm"]:
            self.__btn[i] = QtWidgets.QPushButton(self._central_widget)

    def clear_gtvt_click_pos_3d(self):
        self.__gtvt_click_pos_3d = None

    def clear_gtvn_clicks_pos_3d(self):
        self.__gtvn_clicks_pos_3d = List()

    def _init_data(self, idl_remark: str = None, debug_mode: bool = False):
        super()._init_data()
        self.__debug_mode = debug_mode

        # keep idl.gtvt and idl.gtvn id unchanged
        cur_time = Time.cur_time_str()
        for i in ["gtvt", "gtvn"]:
            self._idl_id[i] = "idl.{}_".format(i) + cur_time
            if debug_mode:
                self._idl_id[i] += "_" + Debug.DELETE_FLAG

            if idl_remark != "" and idl_remark is not None:
                while idl_remark.startswith("_"):
                    idl_remark = idl_remark[1:]
                while idl_remark.endswith("_"):
                    idl_remark = idl_remark[:-1]
                self._idl_id[i] += "_" + idl_remark

        self.__idl_step = Dict()
        for patient in self._patients.to_list():
            self.__idl_step["patient={}".format(patient)] = IDLStep.CLICK_GTVT_CENTER

        # initialize the position of gtvt click / gtvn clicks
        self.clear_gtvt_click_pos_3d()
        self.clear_gtvn_clicks_pos_3d()

        # drawing
        self.drawing_mode = DrawingMode.GTVT_PEN
        self.paint_pos = None  # Store the last painted point
        self.__gtvt_annotated_status = Dict()
        for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
            self.__gtvt_annotated_status[plane] = False

    def __save_idl_step(self):
        for i in ["gtvt", "gtvn"]:
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

    def __clear_all_drawing_layers_on_4_qlabels(self):
        for i in [CT, PT, MR1, MR2]:
            self.img_qlabel[i].drawing_layer = QPixmap(self.img_qlabel[i].size())
            self.img_qlabel[i].drawing_layer.fill(Qt.transparent)
            self.img_qlabel[i].update()

    def _init_side_bar(self):
        super()._init_side_bar()

        # hide replay controls
        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._text_label[i].hide()
            self._combox[i].hide()
            self._arrow_btn["prev.{}".format(i)].hide()
            self._arrow_btn["next.{}".format(i)].hide()

        # show annotation controls
        # self._text_box_annotation_msg.show()
        self._progress_bar_idl.show()
        self._slider["pen.size"].show()
        for i in ["annotation.tools", "idl.progress", "pen.size"]:
            self._text_label[i].show()
        for i in ["pen", "eraser", "clear", "confirm"]:
            self.__btn[i].show()
        for i in ["gtvt", "gtvn"]:
            self._radio_btn["draw.{}".format(i)].show()
            self._radio_btn["draw.{}".format(i)].setFont(self._font_bold)

        self._radio_btn["draw.gtvt"].setChecked(True)

        # set text
        # self._text_box_annotation_msg.setText("Please Select a Patient")
        self._text_label["annotation.tools"].setText("ANNOTATION TOOLS")
        self._text_label["idl.progress"].setText("Retraining Progress")

        # set fonts
        for i in ["annotation.tools", "idl.progress", "pen.size"]:
            self._text_label[i].setFont(self._font_bold)
        self._text_box_annotation_msg.setFont(self._font_bold)

        # set textbox read only
        self._text_box_annotation_msg.setReadOnly(True)

        # pen size slider
        self._slider["pen.size"].setMinimum(1)
        self._slider["pen.size"].setMaximum(7)
        self._slider["pen.size"].setValue(4)

        # set btn icons
        for i in ["pen", "eraser", "clear", "confirm"]:
            icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "{}.png".format(i)))
            if i == "pen":
                self.__btn[i].setIconSize(QSize(24, 24))
            elif i == "eraser":
                self.__btn[i].setIconSize(QSize(31, 31))
            else:
                self.__btn[i].setIconSize(QSize(25, 25))
            self.__btn[i].setIcon(icon)

        # disable all controls
        for i in [
            CT,
            PT,
            MR1,
            MR2,
            TRANSVERSE,
            CORONAL,
            SAGITTAL,
        ]:
            self._radio_btn[i].setEnabled(False)
        for i in ["gtvt", "gtvn"]:
            self._radio_btn["draw.{}".format(i)].setEnabled(False)
        for i in ["pen", "eraser", "clear", "confirm"]:
            self.__btn[i].setEnabled(False)
        for i in ["bright", "contrast"]:
            for j in [CT, PT, MR1, MR2]:
                self._slider["{}.{}".format(i, j)].setEnabled(False)
        for i in ["zoom", "pen.size"]:
            self._slider[i].setEnabled(False)

        # connect ui to functions
        # (put this at the end, because these functions will need the initialization above)
        self.__btn["pen"].clicked.connect(self.__click_btn_pen)
        self.__btn["eraser"].clicked.connect(self.__click_btn_eraser)
        self.__btn["clear"].clicked.connect(self.__click_btn_clear)
        self.__btn["confirm"].clicked.connect(self.__click_btn_confirm)

        self.__btn_group_drawing_mode_gtv = QButtonGroup()
        for i in ["gtvt", "gtvn"]:
            self.__btn_group_drawing_mode_gtv.addButton(
                self._radio_btn["draw.{}".format(i)]
            )
        self.__btn_group_drawing_mode_gtv.buttonClicked.connect(
            self.__switch_drawing_mode_gtv
        )

    def __switch_drawing_mode_gtv(self):
        if self._radio_btn["draw.gtvt"].isChecked():
            if self.drawing_mode == DrawingMode.GTVN_PEN:
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self.drawing_mode == DrawingMode.GTVN_ERASER:
                self.drawing_mode = DrawingMode.GTVT_ERASER
        elif self._radio_btn["draw.gtvn"].isChecked():
            if self.drawing_mode == DrawingMode.GTVT_PEN:
                self.drawing_mode = DrawingMode.GTVN_PEN
            elif self.drawing_mode == DrawingMode.GTVT_ERASER:
                self.drawing_mode = DrawingMode.GTVN_ERASER

    def get_pen_size(self):
        return self._slider["pen.size"].value()

    def _refresh_side_bar(self):
        (
            left,
            top,
            width,
            gap,
            text_height,
            bar_height,
            slider_height,
            radio_btn_height,
        ) = super()._refresh_side_bar(widgets_to_display=["patient"])

        annotation_msg_box_height = 80
        annotation_btn_width = 50
        annotation_btn_height = 40
        radio_btn_width = 90
        radio_btn_gap = 10

        # annotation tools
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label["annotation.tools"].setGeometry(rect)
        self._text_label["annotation.tools"].show()
        top += text_height
        # drawing mode radio btns
        tmp_left = left
        for i in ["gtvt", "gtvn"]:
            rect = QRect(tmp_left, top, radio_btn_width, radio_btn_height)
            self._radio_btn["draw.{}".format(i)].setGeometry(rect)
            tmp_left += radio_btn_gap + radio_btn_width
        top += radio_btn_height
        # annotation buttons
        tmp_left = left
        annotation_btn_gap = round((width - 4 * annotation_btn_width) / 3)
        for i in ["pen", "eraser", "clear", "confirm"]:
            rect = QRect(tmp_left, top, annotation_btn_width, annotation_btn_height)
            self.__btn[i].setGeometry(rect)
            self.__btn[i].show()
            tmp_left += annotation_btn_gap + annotation_btn_width
        top += annotation_btn_height

        # pen size
        rect = QRect(left, top, width, text_height)
        self._text_label["pen.size"].setGeometry(rect)
        top += text_height
        rect = QRect(left, top, width, slider_height)
        self._slider["pen.size"].setGeometry(rect)
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

    def _load_baseline_data(self):
        # self._reset_zoomin()
        self._clear_img_3d()
        self._clear_img_qlabels()

        self._baseline_id = "baseline_real.idl"

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self._combox["patient"].setCurrentIndex(-1)  # show nothing

        # create idl folders (after baseline_id is confirmed)
        for i in ["gtvt", "gtvn"]:
            Dir.create(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i])
            )

    def _add_msg_on_qimg(self, qimg: QImage):
        pos_x = 10
        pos_y = 25

        if self._cur_patient is None:
            text = "Please select a patient"
            self._qimg_draw_text(
                qimg=qimg,
                text=text,
                pos=(pos_x, pos_y),
                color=self._color["green"],
            )
            return

        cur_patient_idl_step = self.get_cur_patient_idl_step()

        if cur_patient_idl_step == IDLStep.CLICK_GTVT_CENTER:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please click the center of primary Gross Tumor Volumes (GTVt)",
                pos=(pos_x, pos_y),
                color=self._color["green"],
            )

        elif cur_patient_idl_step == IDLStep.DRAW_GTVT:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please delineate GTVt in 3 anatomical planes",
                pos=(pos_x, pos_y),
                color=self._color["green"],
            )
            pos_y += 5
            for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
                pos_y += 20
                text = plane.capitalize()
                if self.__gtvt_annotated_status[plane] is True:
                    text += " ✓"
                    color = self._color["green"]
                else:
                    text += " ✕"
                    color = self._color["red"]
                self._qimg_draw_text(
                    qimg=qimg,
                    text=text,
                    pos=(pos_x, pos_y),
                    color=color,
                )

        elif cur_patient_idl_step == IDLStep.CLICK_GTVN_CENTER:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please click the center of malignant lymph nodes (GTVn)",
                pos=(pos_x, pos_y),
                color=self._color["green"],
            )

        elif cur_patient_idl_step == IDLStep.CORRECTION:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please correct the auto-segmentation",
                pos=(pos_x, pos_y),
                color=self._color["green"],
            )

    # rewrite this function (do nothing)
    def _add_score_on_qimg(self, qimg: QImage):
        pass

    def _add_contour_description_on_qimg(
        self,
        qimg: QImage,
        show_user_input_text: bool = False,
    ):
        pos_x = 10
        pos_y = qimg.height() - 13

        for i in ["t", "n"]:
            if self.img_3d["gtv{}.pred".format(i)] is not None:
                text = "GTV{}".format(i)
                self._qimg_draw_text(
                    qimg=qimg,
                    text=text,
                    pos=(pos_x, pos_y),
                    color=self._color["gtv{}.pred".format(i)],
                )
                pos_x += 45

    def _load_patient_data(self, idx: int = None):
        # enable all controls
        for i in [
            CT,
            PT,
            MR1,
            MR2,
            TRANSVERSE,
            CORONAL,
            SAGITTAL,
        ]:
            self._radio_btn[i].setEnabled(True)
        for i in ["clear", "confirm"]:
            self.__btn[i].setEnabled(True)
        for i in ["bright", "contrast"]:
            for j in [CT, PT, MR1, MR2]:
                self._slider["{}.{}".format(i, j)].setEnabled(True)
        for i in ["zoom", "pen.size"]:
            self._slider[i].setEnabled(True)

        self.clear_gtvt_click_pos_3d()
        self.clear_gtvn_clicks_pos_3d()

        self._cur_patient = self._combox["patient"].currentText()
        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_dir_and_nii_spacing()

        # self._reset_zoomin()

        # load multi-modal imgs only, no labels
        self._load_multi_modal_imgs()
        # reset current slice id after ct img loaded
        self._reset_cur_slice_id()
        self._load_idl_gtvt_data()
        self._load_idl_gtvn_data()
        self.refresh_img_qlabels()
        self.__refresh_crosses_on_4_qlabels()
        self.__save_idl_step()

    def _get_middle_slice_id(self):
        if self.img_3d[CT] is None:
            Debug.error_exit("get middle slice id after multi-modal imgs are loaded")

        if self._plane == SAGITTAL:
            slices_count = self.img_3d[CT].shape[2]
        elif self._plane == CORONAL:
            slices_count = self.img_3d[CT].shape[1]
        elif self._plane == TRANSVERSE:
            slices_count = self.img_3d[CT].shape[0]

        if slices_count > 0:
            # show the middle slice of whole 3D img,
            slice_id = round(slices_count / 2) - 1
            slice_id = Value.limit_range(slice_id, (0, slices_count - 1))
            return slice_id
        else:
            return None

    def _reset_cur_slice_id(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER or idl_step == IDLStep.DRAW_GTVT:
            if self.__gtvt_click_pos_3d is None:
                self.cur_slice_id = self._get_middle_slice_id()
            else:
                self.cur_slice_id = self.__get_gtvt_center_slice_id()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            if len(self.__gtvn_clicks_pos_3d) == 0:
                self.cur_slice_id = self.__get_gtvt_center_slice_id()
            else:
                self.cur_slice_id = self.__get_gtvn_center_slice_id()

        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                self.cur_slice_id = self.__get_gtvt_center_slice_id()
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                if len(self.__gtvn_clicks_pos_3d) == 0:
                    self.cur_slice_id = self.__get_gtvt_center_slice_id()
                else:
                    self.cur_slice_id = self.__get_gtvn_center_slice_id()

    def get_cur_patient_idl_step(self):
        return self.__idl_step["patient={}".format(self._cur_patient)]

    def set_cur_patient_idl_step(self, step: str):
        self.__idl_step["patient={}".format(self._cur_patient)] = step

    # def __update_msg(self):
    #     cur_patient_idl_step = self.get_cur_patient_idl_step()

    #     if cur_patient_idl_step == IDLStep.CLICK_GTVT_CENTER:
    #         self._text_box_annotation_msg.setText(
    #             "Please click the center of GTVt, then press OK"
    #         )

    #     elif cur_patient_idl_step == IDLStep.DRAW_GTVT:
    #         self._text_box_annotation_msg.setText(
    #             "Please delineate the countour of GTVt on transvers/coronal/sagittal plane, then press OK"
    #         )

    #     elif cur_patient_idl_step == IDLStep.CLICK_GTVN_CENTER:
    #         self._text_box_annotation_msg.setText(
    #             "Please click the center of each involved lymph nodes, then press OK."
    #         )

    #     elif cur_patient_idl_step == IDLStep.CORRECTION:
    #         self._text_box_annotation_msg.setText(
    #             "Please correct the predictions, then press OK"
    #         )

    def _load_idl_gtvt_data(self):
        self._load_idl_gtv_data(gtv="gtvt")

    def _load_idl_gtvn_data(self):
        self._load_idl_gtv_data(gtv="gtvn")

    def _load_idl_gtv_data(self, gtv: str) -> str:
        round_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id[gtv],
            "patients",
            "patient={}".format(self._cur_patient),
            "round=01",
        )

        nii_name_list = ["pred", "correction", "correction.mask"]
        if gtv == "gtvt":
            nii_name_list += ["click", "annotation"]
        elif gtv == "gtvn":
            nii_name_list.append("clicks")

        for i in nii_name_list:
            nii_path = os.path.join(round_dir, "{}_{}.nii.gz".format(gtv, i))
            if os.path.exists(nii_path):
                self.img_3d["{}.{}".format(gtv, i)] = self._load_3d_img(
                    path=nii_path, binary=True
                )
            else:
                self.img_3d["{}.{}".format(gtv, i)] = None

    def __combine_pred_annotation_correction(self):
        if self.img_3d[CT] is None:
            return

        for i in ["gtvt", "gtvn"]:
            # no pred loaded, generate an empty pred.final
            if self.img_3d["{}.pred".format(i)] is None:
                self.img_3d["{}.pred.final".format(i)] = np.zeros_like(self.img_3d[CT])
            # copy from origin pred
            else:
                self.img_3d["{}.pred.final".format(i)] = self.img_3d[
                    "{}.pred".format(i)
                ].copy()

            # combine gtvt.pred and gtvt.annotation
            if i == "gtvt":
                t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
                self.img_3d["gtvt.pred.final"][t, :, :] = 0
                self.img_3d["gtvt.pred.final"][:, c, :] = 0
                self.img_3d["gtvt.pred.final"][:, :, s] = 0
                self.img_3d["gtvt.pred.final"] = np.maximum(
                    self.img_3d["gtvt.pred.final"], self.img_3d["gtvt.annotation"]
                )

            # combine pred and correction
            if self.img_3d["{}.correction.mask".format(i)] is None:
                continue
            else:
                self.img_3d["{}.pred.final".format(i)] *= (
                    1 - self.img_3d["{}.correction.mask".format(i)]
                )
                self.img_3d["{}.pred.final".format(i)] = np.maximum(
                    self.img_3d["{}.pred.final".format(i)],
                    self.img_3d["{}.correction.mask".format(i)]
                    * self.img_3d["{}.correction".format(i)],
                )
