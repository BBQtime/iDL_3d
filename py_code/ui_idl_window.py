import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import qimage2ndarray
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Timer, Value
from numpy import ndarray
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QEvent, QPoint, QSize, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMessageBox
from scipy import ndimage
from scipy.interpolate import interp1d
from str_lib import DisplayMode, DrawingMode, IDLStep, Modal, Plane
from superqt import QCollapsible
from ui_drag_cross import DragCross
from ui_idl_step_label import IDLStepLabel
from ui_idl_thread import IDLGTVnThread, IDLGTVtThread
from ui_img_frame import ImgFrame
from ui_replay_window import ReplayWindow


class IDLTimer:
    def __init__(
        self,
        baseline_id: str,
        idl_gtvt_id: str,
        patient: str,
        idl_step: str,
    ):
        self.__start_time = None
        self.__patient = patient
        self.__idl_step = idl_step
        # json path
        self.__json_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            baseline_id,
            idl_gtvt_id,
            "time_used.json",
        )
        # create an empty json file
        if not os.path.exists(self.__json_path):
            Json.save({}, self.__json_path)

    def start(self):
        self.__start_time = datetime.now()

    def end(self):
        if self.__start_time is None:
            return

        duration = datetime.now() - self.__start_time
        self.__start_time = None
        total_seconds = int(duration.total_seconds())
        # create a new timedelta without microseconds
        duration = timedelta(seconds=total_seconds)
        duration = str(duration)
        # save json
        time_log = Json.load(self.__json_path)
        time_log["patient={}".format(self.__patient)][self.__idl_step] = duration
        Json.save(time_log, self.__json_path)


class IDLWindow(ReplayWindow):
    def __init__(
        self,
        user_name: str,
        train_id: str,
        debug_mode: bool = False,
    ):
        self.__user_name = user_name
        self.__train_id = train_id
        # pass debug_mode parameter to the parent class
        super().__init__()

    def draw_on_img_frame_press(self, event: QtGui.QMouseEvent, img_frame: ImgFrame):
        if self.cur_idl_step not in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            return

        # switch to the gtvt center slice if current slice is not.
        if self.cur_idl_step == IDLStep.DRAW_GTVT:
            gtvt_center_slice_id = self.__get_gtvt_center_slices_id()[img_frame.plane]
            if self.cur_slice_id[img_frame.plane] != gtvt_center_slice_id:
                self.cur_slice_id[img_frame.plane] = gtvt_center_slice_id
                # (1) PLANE_FIXED mode, only refresh current img frame
                if self.display_mode() == DisplayMode.PLANE_FIXED:
                    frame_name = img_frame.get_frame_name()
                    self.refresh_imgs(frame_name=frame_name)
                # (2) MODAL_FIXED mode, refresh all 4 img frames
                else:
                    self.refresh_imgs()
                return

        # (1) for pen/eraser mode, record paint position
        if self.drawing_mode in [
            DrawingMode.GTVT_PEN,
            DrawingMode.GTVN_PEN,
            DrawingMode.GTVT_ERASER,
            DrawingMode.GTVN_ERASER,
        ]:
            self.paint_pos = event.pos()
            return

        # (2) clear gtvt delineation in current plane
        elif self.cur_idl_step == IDLStep.DRAW_GTVT and self.drawing_mode in [
            DrawingMode.GTVT_CLEAR,
            DrawingMode.GTVN_CLEAR,
        ]:
            # get the position of gtvt center
            d, h, w = np.where(self.img_3d["gtvt.click"] == 1)
            d, h, w = int(d), int(h), int(w)

            delineation = self.img_3d["gtvt.delineation"]

            # clear delineation on cur plane
            if img_frame.plane == Plane.TRANSVERSE:
                delineation[d, :, :] = np.zeros_like(delineation[d, :, :])
            elif img_frame.plane == Plane.CORONAL:
                delineation[:, h, :] = np.zeros_like(delineation[:, h, :])
            elif img_frame.plane == Plane.SAGITTAL:
                delineation[:, :, w] = np.zeros_like(delineation[:, :, w])

            # update gtvt delineated state
            self.__gtvt_delineated_state[img_frame.plane] = False
            # update todo list
            self._text_label[
                "draw.gtvt.{}".format(img_frame.plane)
            ].set_status_missing()

            # refresh contours only (on all img frames) after using "clear" tool
            self.refresh_imgs(
                reload_origin_rgb=False,
                reload_zoomed_rgb=False,
            )

        # (3) "clear" / "restore" in correction step
        elif self.cur_idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            if self.drawing_mode in [
                DrawingMode.GTVT_CLEAR,
                DrawingMode.GTVT_RESTORE,
            ]:
                gtv = "gtvt"
            elif self.drawing_mode in [
                DrawingMode.GTVN_CLEAR,
                DrawingMode.GTVN_RESTORE,
            ]:
                gtv = "gtvn"

            d = h = w = self.cur_slice_id[img_frame.plane]
            correction = self.img_3d["{}.correction".format(gtv)]
            correction_mask = self.img_3d["{}.correction.mask".format(gtv)]

            if img_frame.plane == Plane.TRANSVERSE:
                if self.drawing_mode in [
                    DrawingMode.GTVT_CLEAR,
                    DrawingMode.GTVN_CLEAR,
                ]:
                    # clear correction and enable correction mask
                    correction[d, :, :] = np.zeros_like(correction[d, :, :])
                    correction_mask[d, :, :] = np.ones_like(correction[d, :, :])
                    self.__interpolation(
                        cur_slice_id=d,
                        correction=correction,
                        correction_mask=correction_mask,
                        pred_final=self.img_3d["{}.pred.final".format(gtv)],
                    )

                if self.drawing_mode in [
                    DrawingMode.GTVT_RESTORE,
                    DrawingMode.GTVN_RESTORE,
                ]:
                    # clear slices within interpolation_step
                    slice_id_list = [d]
                    if self.interpolation_step > 1:
                        for i in range(1, self.interpolation_step - 1):
                            slice_id_list.append(d + i)
                            slice_id_list.append(d - i)
                    for i in slice_id_list:
                        if i < 0 or i > correction.shape[0]:
                            continue
                        else:
                            # clear both correction and mask
                            correction[i, :, :] = np.zeros_like(correction[i, :, :])
                            correction_mask[i, :, :] = np.zeros_like(
                                correction_mask[i, :, :]
                            )

            elif img_frame.plane == Plane.CORONAL:
                # clear correction
                correction[:, h, :] = np.zeros_like(correction[:, h, :])
                # "clear" mode, enable correction mask
                if self.drawing_mode in [
                    DrawingMode.GTVT_CLEAR,
                    DrawingMode.GTVN_CLEAR,
                ]:
                    correction_mask[:, h, :] = np.ones_like(correction_mask[:, h, :])
                # "restore" mode, disable correction mask
                elif self.drawing_mode in [
                    DrawingMode.GTVT_RESTORE,
                    DrawingMode.GTVN_RESTORE,
                ]:
                    correction_mask[:, h, :] = np.zeros_like(correction_mask[:, h, :])

            elif img_frame.plane == Plane.SAGITTAL:
                # clear correction
                correction[:, :, w] = np.zeros_like(correction[:, :, w])
                # "clear" mode, enable correction mask
                if self.drawing_mode in [
                    DrawingMode.GTVT_CLEAR,
                    DrawingMode.GTVN_CLEAR,
                ]:
                    correction_mask[:, :, w] = np.ones_like(correction_mask[:, :, w])
                # "restore" mode, disable correction mask
                elif self.drawing_mode in [
                    DrawingMode.GTVT_RESTORE,
                    DrawingMode.GTVN_RESTORE,
                ]:
                    correction_mask[:, :, w] = np.zeros_like(correction_mask[:, :, w])

            # update 3d np arrays
            self.__combine_pred_delineation_correction()
            # refresh contours only (on all img frames) after using "clear" tool
            self.refresh_imgs(
                reload_origin_rgb=False,
                reload_zoomed_rgb=False,
            )

    def draw_on_img_frame_move(self, event: QtGui.QMouseEvent, img_frame: ImgFrame):
        if self.paint_pos is None:
            return

        pen_size = self.get_pen_size()
        eraser_size = self.get_eraser_size()
        eraser_color = QtGui.QColor(*self.color["eraser"])

        if self.display_mode() == DisplayMode.MODAL_FIXED:
            frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            frame_name_list = [img_frame.plane]
        for i in frame_name_list:
            painter = QtGui.QPainter(self.img_frame[i].drawing_layer)

            # transparent pen
            # if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            #     painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            #     painter.setPen(
            #         QtGui.QPen(Qt.transparent, pen_size + 2, Qt.SolidLine, Qt.RoundCap)
            #     )

            # smooth
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            # Set the composition mode to control alpha blending
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

            # delineate gtvt
            if self.cur_idl_step == IDLStep.DRAW_GTVT:
                pen_color = QtGui.QColor(*self.color["gtvt.delineation"])
            # correct gtvt/gtvn
            else:
                if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                    pen_color = QtGui.QColor(*self.color["gtvt.pred"])
                elif self.drawing_mode in [
                    DrawingMode.GTVN_PEN,
                    DrawingMode.GTVN_ERASER,
                ]:
                    pen_color = QtGui.QColor(*self.color["gtvn.pred"])

            if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
                painter.setPen(
                    QtGui.QPen(eraser_color, eraser_size, Qt.SolidLine, Qt.RoundCap)
                )
                self.img_frame[i].pen_mode = False
            elif self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
                painter.setPen(
                    QtGui.QPen(pen_color, pen_size, Qt.SolidLine, Qt.RoundCap)
                )
                self.img_frame[i].pen_mode = True

            painter.drawLine(self.paint_pos, event.pos())

            self.img_frame[i].update()  # schedule a repaint

        self.paint_pos = event.pos()  # update paint pos

    def get_zoomed_rgb_shape(self, frame_name: str):
        return self._zoomed_rgb[frame_name][:, :, 0].shape

    def get_origin_rgb_shape(self, frame_name: str):
        return self._origin_rgb[frame_name][:, :, 0].shape

    def draw_on_img_frame_release(self, img_frame: ImgFrame):
        if self.paint_pos is None:
            return

        # get img frame name
        frame_name = img_frame.get_frame_name()

        # binarize threshold
        # this is for saving qimage as ndarray
        # binarization is needed before and after resize the ndarray
        binary_threshold = 0.5

        # save drawing layer into 2d ndarray
        # qpixmap to a qimage
        qimg = img_frame.drawing_layer.toImage()
        # qimage to ndarray
        new_drawing = qimage2ndarray.alpha_view(qimg).astype(np.float32)
        new_drawing /= 255
        # binarization (before resize)
        new_drawing = Img.binarize(img=new_drawing, threshold=binary_threshold)

        # overwrite delineation back to zoomed_img
        empty_zoomed_img = np.zeros_like(self._zoomed_rgb[frame_name][:, :, 0])
        center_x_pct, center_y_pct = img_frame.img_center_pct
        frame_h, frame_w = new_drawing.shape
        zoomed_h, zoomed_w = empty_zoomed_img.shape
        # Calculate original intended center positions
        center_x_abs = int(round(zoomed_w * center_x_pct))
        center_y_abs = int(round(zoomed_h * center_y_pct))
        # Calculate the amount of black border added to each side
        border_w = max(0, (frame_w - zoomed_w) // 2)
        border_h = max(0, (frame_h - zoomed_h) // 2)
        # Calculate start positions for placing new_drawing
        start_x = center_x_abs - math.ceil(frame_w / 2)
        start_x = max(0, start_x)
        start_y = center_y_abs - math.ceil(frame_h / 2)
        start_y = max(0, start_y)
        # Calculate end positions, do not exceed the zoomed image dimensions
        end_x = start_x + frame_w
        end_y = start_y + frame_h
        end_x = min(end_x, zoomed_w)
        end_y = min(end_y, zoomed_h)
        # Calculate crop dimensions
        crop_w = frame_w - border_w * 2
        crop_h = frame_h - border_h * 2
        # Ensure new_drawing cropped area does not exceed zoomed img
        crop_w = min(crop_w, zoomed_w)
        crop_h = min(crop_h, zoomed_h)
        # crop new_drawing
        new_drawing = new_drawing[
            border_h : crop_h + border_h, border_w : crop_w + border_w
        ]
        # Place the new_drawing onto the adjusted position within empty_zoomed_img
        empty_zoomed_img[start_y:end_y, start_x:end_x] = new_drawing
        new_drawing = empty_zoomed_img

        # resize to origin img size (slice from 3d img)
        if img_frame.plane == Plane.SAGITTAL:
            actual_shape = self.img_3d[Modal.CT][:, :, 0].shape
        elif img_frame.plane == Plane.CORONAL:
            actual_shape = self.img_3d[Modal.CT][:, 0, :].shape
        elif img_frame.plane == Plane.TRANSVERSE:
            actual_shape = self.img_3d[Modal.CT][0, :, :].shape
        new_drawing = cv2.resize(
            new_drawing,
            (actual_shape[1], actual_shape[0]),
            interpolation=cv2.INTER_AREA,  # best for scaling down
        )
        # binarization (after resize)
        new_drawing = Img.binarize(img=new_drawing, threshold=binary_threshold)

        # add 2d delineation/correction on 3d ndarray

        # (1) gtvt delineation
        if self.cur_idl_step == IDLStep.DRAW_GTVT:
            d, h, w = self.gtvt_click_pos_3d
            if img_frame.plane == Plane.TRANSVERSE:
                exist_drawing = self.img_3d["gtvt.delineation"][d, :, :]
            elif img_frame.plane == Plane.CORONAL:
                exist_drawing = self.img_3d["gtvt.delineation"][:, h, :]
            elif img_frame.plane == Plane.SAGITTAL:
                exist_drawing = self.img_3d["gtvt.delineation"][:, :, w]
        # (2) gtvt/gtvn correction
        elif self.cur_idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            d = h = w = self.cur_slice_id[img_frame.plane]
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
            pred_final = self.img_3d["{}.pred.final".format(gtv)]
            # copy from gtvt/gtvn.pred.final, dont change original data
            if img_frame.plane == Plane.TRANSVERSE:
                exist_drawing = pred_final[d, :, :].copy()
            elif img_frame.plane == Plane.CORONAL:
                exist_drawing = pred_final[:, h, :].copy()
            elif img_frame.plane == Plane.SAGITTAL:
                exist_drawing = pred_final[:, :, w].copy()

        # invert color if in eraser mode
        if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            exist_drawing = 1 - exist_drawing

        # combine new_drawing and new_drawing
        new_drawing = np.maximum(exist_drawing, new_drawing)

        # fill holes if in pen mode
        if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
            new_drawing = ndimage.binary_fill_holes(new_drawing).astype(np.float32)

        # invert color back, if in eraser mode
        if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            new_drawing = 1 - new_drawing

        # DRAW_GTVT mode
        if self.cur_idl_step == IDLStep.DRAW_GTVT:
            delineation = self.img_3d["gtvt.delineation"]
            # replace slice in 3d delineation
            if img_frame.plane == Plane.TRANSVERSE:
                delineation[d, :, :] = new_drawing
            elif img_frame.plane == Plane.CORONAL:
                delineation[:, h, :] = new_drawing
            elif img_frame.plane == Plane.SAGITTAL:
                delineation[:, :, w] = new_drawing

        elif self.cur_idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                correction = self.img_3d["gtvt.correction"]
                correction_mask = self.img_3d["gtvt.correction.mask"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                correction = self.img_3d["gtvn.correction"]
                correction_mask = self.img_3d["gtvn.correction.mask"]

            # replace slice in 3d correction
            if img_frame.plane == Plane.TRANSVERSE:
                correction[d, :, :] = new_drawing
            elif img_frame.plane == Plane.CORONAL:
                correction[:, h, :] = new_drawing
            elif img_frame.plane == Plane.SAGITTAL:
                correction[:, :, w] = new_drawing

            if img_frame.plane == Plane.TRANSVERSE:
                if new_drawing.max() == 0:
                    correction_mask[d, :, :] = np.zeros_like(new_drawing)
                else:
                    correction_mask[d, :, :] = np.ones_like(new_drawing)
                self.__interpolation(
                    cur_slice_id=d,
                    correction=correction,
                    correction_mask=correction_mask,
                    pred_final=pred_final,
                )

            elif img_frame.plane == Plane.CORONAL:
                if new_drawing.max() == 0:
                    correction_mask[:, h, :] = np.zeros_like(new_drawing)
                else:
                    correction_mask[:, h, :] = np.ones_like(new_drawing)

            elif img_frame.plane == Plane.SAGITTAL:
                if new_drawing.max() == 0:
                    correction_mask[:, :, w] = np.zeros_like(new_drawing)
                else:
                    correction_mask[:, :, w] = np.ones_like(new_drawing)

        # update values
        self.paint_pos = None
        self.__update_gtvt_delineated_status()
        self.__combine_pred_delineation_correction()

        # update UI
        self.__clear_all_drawing_layers(img_frame)
        # refresh contours only (on all img frames) after drawing
        self.refresh_imgs(
            reload_origin_rgb=False,
            reload_zoomed_rgb=False,
        )

    def __interpolation(
        self,
        cur_slice_id: int,
        correction: ndarray,
        correction_mask: ndarray,
        pred_final: ndarray,
    ):
        start_end_slices_pairs = []

        # prev slice id
        prev_slice_id = cur_slice_id - self.interpolation_step
        prev_slice_id = max(prev_slice_id, 0)
        if cur_slice_id > prev_slice_id:
            start_end_slices_pairs.append((prev_slice_id, cur_slice_id))

        # next slice id
        next_slice_id = cur_slice_id + self.interpolation_step
        next_slice_id = min(next_slice_id, correction.shape[0])
        if cur_slice_id < next_slice_id:
            start_end_slices_pairs.append((cur_slice_id, next_slice_id))

        for start_slice_id, end_slice_id in start_end_slices_pairs:
            if start_slice_id == cur_slice_id:
                start_slice_data = correction[start_slice_id, :, :]
                end_slice_data = pred_final[end_slice_id, :, :]
            else:
                start_slice_data = pred_final[start_slice_id, :, :]
                end_slice_data = correction[end_slice_id, :, :]

            # generate interpolated slices
            interpolation_func = interp1d(
                [start_slice_id, end_slice_id],
                np.array([start_slice_data, end_slice_data]),
                axis=0,
                kind="slinear",
            )
            interpolated_slices = interpolation_func(
                np.arange(start_slice_id + 1, end_slice_id)
            )
            interpolated_slices = Img.binarize(interpolated_slices)

            # add interpolated slices
            correction[start_slice_id + 1 : end_slice_id, :, :] = interpolated_slices

            # update correction mask for interpolated slices
            for i in range(start_slice_id + 1, end_slice_id):
                correction_mask[i, :, :] = np.ones_like(correction[i, :, :])

    def __save_corrections_and_masks(self):
        for gtv in ["gtvt", "gtvn"]:

            if self.img_3d["{}.correction".format(gtv)] is None:
                continue

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
                        cur_round_dir,
                        "{}_{}.nii.gz".format(
                            gtv,
                            # "correction.mask" -> "correction_mask"
                            i.replace(".", "_"),
                        ),
                    ),
                    spacing=self._nii_spacing,
                )

    # this function is connected to widget, dont set input params to this function
    def __on_btn_pen_clicked(self):

        # (1) update drawing mode
        if self.cur_idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECT_GTVT]:
            self.drawing_mode = DrawingMode.GTVT_PEN
        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            self.drawing_mode = DrawingMode.GTVN_PEN
        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_PEN

        # (2) update widgets
        if self.cur_idl_step in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self._text_label["eraser.size"].hide()
            self._slider["eraser.size"].hide()
            self._text_label["pen.size"].show()
            self._slider["pen.size"].show()

    def _init_widgets_todo_list(self):
        idl_step_list = [
            IDLStep.SELECT_PATIENT,
            IDLStep.CLICK_GTVT_CENTER,
            IDLStep.DRAW_GTVT,
            IDLStep.DRAW_GTVT_TRANSVERSE,
            IDLStep.DRAW_GTVT_CORONAL,
            IDLStep.DRAW_GTVT_SAGITTAL,
            IDLStep.CLICK_GTVN_CENTER,
            IDLStep.WAITING,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
        ]

        # init IDLStepLabel
        for i in idl_step_list:
            # create idl step label
            self._text_label[i] = IDLStepLabel(idl_step=i)
            # set init state
            if i == IDLStep.SELECT_PATIENT:
                self._text_label[i].set_status_ongoing()
            else:
                self._text_label[i].set_status_notstart()

        # button size 562*187
        btn_h = 27 if g.is_linux() else 40
        btn_w = round(btn_h * 562 / 187)
        self._btn["next.step"] = QtWidgets.QPushButton()
        self._btn["next.step"].setFixedSize(QSize(btn_w, btn_h))
        self._btn["next.step"].clicked.connect(self.__on_btn_next_step_clicked)
        # set btn icons
        pixmap = QtGui.QPixmap(os.path.join(g.PROJ_DIR, "icons", "next_step.png"))
        # pixmap = pixmap.scaled(100, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = pixmap.scaled(
            btn_w, btn_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        icon = QtGui.QIcon(pixmap)
        self._btn["next.step"].setIconSize(QSize(btn_w, btn_h))
        self._btn["next.step"].setIcon(icon)
        self._btn["next.step"].setStyleSheet(
            "QPushButton { border: none; margin: 0px; padding: 0px; }"
        )

        # v layout
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.setSpacing(2 if g.is_linux() else 2)
        v_layout.addWidget(
            self._btn["next.step"], alignment=Qt.AlignmentFlag.AlignRight
        )
        for i in self._text_label.keys():
            if i in idl_step_list:
                v_layout.addWidget(self._text_label[i])

        # container
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)

        # create qcollapsible space
        self._collap["todo.list"] = QCollapsible("TODO LIST")
        self._collap["todo.list"].addWidget(container)
        self._collap["todo.list"].expand()

    def __clear_3d_imgs_and_delete_nii(self, img_name_list: list):
        # clear 3d imgs
        for i in img_name_list:
            self.img_3d[i] = None

        # delete nii files
        for gtv in ["gtvt", "gtvn"]:
            imgs_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id[gtv],
                "patients",
                "patient={}".format(self._cur_patient),
                "round=01",
            )
            for i in img_name_list:
                # delete nii file
                if i.startswith(gtv):
                    file_name = i.replace(".", "_") + ".nii.gz"
                    Dir.delete(os.path.join(imgs_dir, file_name))

                # delete idl.gtvn related files
                if i == "gtvn.pred":
                    Dir.delete(os.path.join(imgs_dir, "gtvn_distance_map.nii.gz"))
                    # fold_dirs = Dir.get_sub_dirs(
                    #     input_dir=os.path.join(
                    #         g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id["gtvn"]
                    #     ),
                    #     key_word="fold=",
                    #     full_path=True,
                    # )
                    # for fold_dir in fold_dirs:
                    #     Dir.delete(fold_dir)

                # delete idl.gtvt related files
                if i == "gtvt.pred":
                    Dir.delete(os.path.join(imgs_dir, "round=01.pt"))
                    Dir.delete(os.path.join(Path(imgs_dir).parent, "loss.json"))
                    Dir.delete(
                        os.path.join(Path(imgs_dir).parent, "selected_slices.json")
                    )
                    # Dir.delete(
                    #     os.path.join(Path(imgs_dir).parent.parent.parent, "hyper.json")
                    # )
                    Dir.delete(
                        os.path.join(Path(imgs_dir).parent.parent.parent, "loss.png")
                    )

    def __goto_idl_step_click_gtvt_center(self):
        # (1) stop idl qthreads
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # (2) update status
        self.__update_cur_idl_step(IDLStep.CLICK_GTVT_CENTER)

        # (3) clear gtvt click pos
        self.gtvt_click_pos_3d = None
        # DO NOT clear self.gtvn_clicks_pos_3d

        # (4) clear 3d imgs and delete nii files
        img_name_list = ["gtvt.click", "gtvn.clicks", "gtvt.delineation"]
        for i in ["gtvt", "gtvn"]:
            img_name_list += [
                "{}.pred".format(i),
                "{}.correction".format(i),
                "{}.correction.mask".format(i),
                "{}.pred.final".format(i),
            ]
        self.__clear_3d_imgs_and_delete_nii(img_name_list)

        # (5) refresh todolist, imgs and crosses
        self.__refresh_todo_list()
        self.refresh_imgs()
        self.refresh_crosses()

        # (6) update widgets
        self.restore_mouse_cursor()
        self.__disable_annotation_tools()
        self._btn["next.step"].setEnabled(True)

        # (7) start recording time
        self.__timer[IDLStep.CLICK_GTVT_CENTER].start()

    def __confirm_gtvt_center(self):
        # (1) check if there is gtvt click
        if self.gtvt_click_pos_3d is None:
            QMessageBox.information(
                self,
                "Information",
                "GTVt center not detected.",
                QMessageBox.Ok,
            )
            return

        # (2) add clicks into 3d img
        pos = self.gtvt_click_pos_3d
        if self.img_3d["gtvt.click"] is None:
            self.img_3d["gtvt.click"] = np.zeros_like(self.img_3d[Modal.CT])
        # pos 0-transverse 1-coronal 2-saggital
        self.img_3d["gtvt.click"][pos[0]][pos[1]][pos[2]] = 1

        # (3) save gtvt_click
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

        # (4) save gtvt selected_slices.json
        pos = np.where(idl_gtvt_click == 1)
        selected_slices = Dict()
        selected_slices[Plane.TRANSVERSE]["round=01"] = List(pos[0]).to_str()
        selected_slices[Plane.CORONAL]["round=01"] = List(pos[1]).to_str()
        selected_slices[Plane.SAGITTAL]["round=01"] = List(pos[2]).to_str()
        Json.save(
            data=selected_slices,
            path=os.path.join(cur_patient_dir, "selected_slices.json"),
        )

        # (5) end timing
        self.__timer[IDLStep.CLICK_GTVT_CENTER].end()

        # (6) goto next step
        self.__goto_idl_step_draw_gtvt()

    def __goto_idl_step_draw_gtvt(self):
        # (1) stop idl qthread
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # (2) update status
        self.__update_cur_idl_step(IDLStep.DRAW_GTVT)
        self.drawing_mode = DrawingMode.GTVT_PEN

        # DO NOT clear self.gtvt_click_pos_3d
        # DO NOT clear self.gtvn_clicks_pos_3d

        # (3) clear 3d imgs and delete nii files
        img_name_list = ["gtvt.delineation", "gtvn.clicks"]
        for i in ["gtvt", "gtvn"]:
            img_name_list += [
                "{}.pred".format(i),
                "{}.correction".format(i),
                "{}.correction.mask".format(i),
                "{}.pred.final".format(i),
            ]
        self.__clear_3d_imgs_and_delete_nii(img_name_list)
        self.img_3d["gtvt.delineation"] = np.zeros_like(self.img_3d[Modal.CT])

        # (4) refresh todolist and imgs, delete crosses
        self.__refresh_todo_list()
        self.refresh_imgs()  # after __refresh_todo_list()
        self.delete_all_crosses()

        # (5) update widgets
        self.__enable_annotation_tools()
        self._btn["next.step"].setEnabled(True)

        # (6) start recording time
        self.__timer[IDLStep.DRAW_GTVT].start()

    def __confirm_gtvt_delineation(self):
        # (1) check if gtvt are delineated in 3 planes
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if self.__gtvt_delineated_state[plane] is False:
                QMessageBox.information(
                    self,
                    "Information",
                    "Please draw GTVt in {} plane.".format(plane),
                    QMessageBox.Ok,
                )
                self._modal_fixed_mode_switch_plane(plane)
                if self.drawing_mode == DrawingMode.GTVT_ERASER:
                    self.drawing_mode = DrawingMode.GTVT_PEN
                    self._text_label["eraser.size"].hide()
                    self._slider["eraser.size"].hide()
                    self._text_label["pen.size"].show()
                    self._slider["pen.size"].show()
                return

        # (2) save gtvt delineation
        cur_round_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id["gtvt"],
            "patients",
            "patient={}".format(self._cur_patient),
            "round=01",
        )
        Dir.create(cur_round_dir)
        gtvt_delineation_to_save = self.img_3d["gtvt.delineation"].copy()
        # flip left/right for 1mm data
        if self._nii_spacing[2] == 1.0:
            gtvt_delineation_to_save = np.flip(gtvt_delineation_to_save, axis=2)
        # turn upside down
        gtvt_delineation_to_save = np.flip(gtvt_delineation_to_save, axis=0)
        Nii.save(
            img=gtvt_delineation_to_save,
            save_path=os.path.join(cur_round_dir, "gtvt_delineation.nii.gz"),
            spacing=self._nii_spacing,
        )

        # (3) start idl gtvt thread
        self.__idl_gtvt_thread.set_param(
            idl_gtvt_id=self._idl_id["gtvt"],
            patient=self._cur_patient,
            dataset_ver=self._dataset_ver,
            debug_mode=self._debug_mode,
        )
        self.__idl_gtvt_thread.start()

        # (4) end and start timer
        self.__timer[IDLStep.DRAW_GTVT].end()
        self.__timer[IDLStep.WAITING_GTVT].start()

        # (5) goto next step
        self.__goto_idl_step_click_gtvn_center()

    def __goto_idl_step_click_gtvn_center(self):
        # (1) stop idl gtvn qthread
        self.__idl_gtvn_thread.stop()

        # (2) update status (before refresh images)
        self.__update_cur_idl_step(IDLStep.CLICK_GTVN_CENTER)

        # (3) clear gtvn clicks
        self.gtvn_clicks_pos_3d = List()

        # (4) clear 3d imgs and delete nii files, then combine images
        img_name_list = [
            "gtvn.clicks",
            "gtvn.pred",
            "gtvn.correction",
            "gtvn.correction.mask",
            "gtvn.pred.final",
        ]
        self.__clear_3d_imgs_and_delete_nii(img_name_list)
        self.__combine_pred_delineation_correction()

        # (5) refresh todolist, imgs and crosses
        self.__refresh_todo_list()
        self.refresh_imgs()
        self.refresh_crosses()

        # (6) update widgets
        self.restore_mouse_cursor()
        self.__disable_annotation_tools()
        self._btn["next.step"].setEnabled(True)

        # (7) start recording time
        self.__timer[IDLStep.CLICK_GTVN_CENTER].start()

    def __goto_idl_step_correct_both(self, drawing_mode: str = DrawingMode.GTVT_PEN):
        # (1) stop idl qthreads
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # (2) update status
        self.__update_cur_idl_step(IDLStep.CORRECT_BOTH)
        self.drawing_mode = drawing_mode

        # (3) update imgs
        # init correction and mask
        for gtv in ["gtvt", "gtvn"]:
            for i in ["{}.correction".format(gtv), "{}.correction.mask".format(gtv)]:
                if self.img_3d[i] is None:
                    self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])
        self.__combine_pred_delineation_correction()

        # (4) refresh todolist and imgs
        self.__refresh_todo_list()
        self.refresh_imgs()

        # (5) update widgets
        if self.drawing_mode in [
            DrawingMode.GTVT_PEN,
            DrawingMode.GTVT_ERASER,
            DrawingMode.GTVT_CLEAR,
            DrawingMode.GTVT_RESTORE,
        ]:
            self._radio_btn["correct.gtvt"].setChecked(True)
        elif self.drawing_mode in [
            DrawingMode.GTVN_PEN,
            DrawingMode.GTVN_ERASER,
            DrawingMode.GTVN_CLEAR,
            DrawingMode.GTVN_RESTORE,
        ]:
            self._radio_btn["correct.gtvn"].setChecked(True)
        self.__enable_annotation_tools()
        self._btn["next.step"].setEnabled(True)

        # (6) start timer
        self.__timer[IDLStep.CORRECT_GTVT].start()
        self.__timer[IDLStep.CORRECT_GTVN].start()

    def __goto_idl_step_approved(self):
        # (1) stop idl qthreads (if running)
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # (2) update status
        self.__update_cur_idl_step(IDLStep.APPROVED)

        # (3) update imgs
        self.__combine_pred_delineation_correction()

        # (4) refresh todolist and imgs
        self.__refresh_todo_list()
        self.refresh_imgs()

        # (5) update widgets
        self.restore_mouse_cursor()
        self.__disable_annotation_tools()
        self._btn["next.step"].setEnabled(False)
        self._collap["annotation"].collapse()
        self._collap["patient"].expand()

        # (6) end timers
        self.__timer[IDLStep.CORRECT_GTVT].end()
        self.__timer[IDLStep.CORRECT_GTVN].end()

    def on_idl_step_text_box_clicked(self, text_box: IDLStepLabel):
        # (1) jump to step SELECT_PATIENT
        if text_box == self._text_label[IDLStep.SELECT_PATIENT]:
            if not self._collap["patient"].isExpanded():
                self._collap["patient"].expand()
            # expande combobox patient, simulate click
            # dont use "QCombobox.showPopup()", this will set focus to "QListView"
            # when mouse is released, QListView will disappear
            event = QMouseEvent(
                QEvent.MouseButtonPress,
                QPoint(0, 0),
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier,
            )
            QtWidgets.QApplication.postEvent(self.combox["patient"], event)

        # (2) jump to step CLICK_GTVT_CENTER
        elif text_box == self._text_label[IDLStep.CLICK_GTVT_CENTER]:
            if self.cur_idl_step not in [
                IDLStep.DRAW_GTVT,
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
                IDLStep.CORRECT_BOTH,
                IDLStep.APPROVED,
            ]:
                return

            text = (
                "Would you like to revert to SETP 2 and re-click the center of GTVt? "
                "This will clear all your previous GTVt delineations and corrections "
                "and the neural network will need to regenerate the segmentation."
            )
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

            self.__goto_idl_step_click_gtvt_center()

        # (3) jump to step DRAW_GTVT
        elif text_box in [
            self._text_label[IDLStep.DRAW_GTVT],
            self._text_label[IDLStep.DRAW_GTVT_TRANSVERSE],
            self._text_label[IDLStep.DRAW_GTVT_CORONAL],
            self._text_label[IDLStep.DRAW_GTVT_SAGITTAL],
        ]:
            if self.cur_idl_step not in [
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
                IDLStep.CORRECT_BOTH,
                IDLStep.APPROVED,
            ]:
                return

            text = (
                "Would you like to revert to SETP 3 and re-delineate GTVt? "
                "This will clear all your previous GTVt delineations and corrections "
                "and the neural network will need to regenerate the segmentation."
            )
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

            self.__goto_idl_step_draw_gtvt()

        # (4) jump to step CLICK_GTVN_CENTER
        elif text_box == self._text_label[IDLStep.CLICK_GTVN_CENTER]:
            if self.cur_idl_step not in [
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
                IDLStep.CORRECT_BOTH,
                IDLStep.APPROVED,
            ]:
                return

            text = (
                "Would you like to revert to SETP 4 and re-click the centers of GTVn? "
                "This will clear all your previous GTVn clicks and corrections "
                "and the neural network will need to regenerate the segmentation."
            )
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

            self.__goto_idl_step_click_gtvn_center()

        # (5) revert to step CORRECT_GTVT/GTVN
        elif text_box in [
            self._text_label[IDLStep.CORRECT_GTVT],
            self._text_label[IDLStep.CORRECT_GTVN],
        ]:
            if self.cur_idl_step != IDLStep.APPROVED:
                return

            text = "Would you like to revert to SETP 6 and re-correct the predictions?"
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

            # goto idl step correct both
            if text_box == self._text_label[IDLStep.CORRECT_GTVT]:
                self.__goto_idl_step_correct_both(DrawingMode.GTVT_PEN)
            else:
                self.__goto_idl_step_correct_both(DrawingMode.GTVN_PEN)
            self._collap["patient"].collapse()

    # this function is connected to widget, dont set input params to this function
    def __on_btn_eraser_clicked(self):
        # (1) update drawing mode
        if self.cur_idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECT_GTVT]:
            self.drawing_mode = DrawingMode.GTVT_ERASER
        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            self.drawing_mode = DrawingMode.GTVN_ERASER
        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_ERASER
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_ERASER

        # (2) update widgets
        if self.cur_idl_step in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self._text_label["pen.size"].hide()
            self._slider["pen.size"].hide()
            self._text_label["eraser.size"].show()
            self._slider["eraser.size"].show()

    def __update_idl_gtvt_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        Value.limit_range(progress_int, (0, 100))
        self.__progress_bar["gtvt"].setValue(progress_int)

    def __on_idl_gtvt_thread_finished(self):
        # (1) update status
        # dont update idl step if user has not submitted gtvn clicks
        if self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            pass
        elif self.__idl_gtvn_thread.is_running:
            self.__update_cur_idl_step(IDLStep.CORRECT_GTVT)
        else:
            self.__update_cur_idl_step(IDLStep.CORRECT_BOTH)

        # (2) load and combine 3d imgs
        self._load_idl_gtvt_data()
        self.__combine_pred_delineation_correction()
        # init correction and mask
        # (they are empty anyway, its efficient to init them after __combine_pred_delineation_correction)
        for i in ["gtvt.correction", "gtvt.correction.mask"]:
            self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])

        # (3) refresh todolist and imgs
        self.__refresh_todo_list()
        # refresh contours only (on all img frames)
        self.refresh_imgs(
            reload_origin_rgb=False,
            reload_zoomed_rgb=False,
        )

        # (4) update widgets
        self.__enable_annotation_tools()
        # (4-1) CORRECT_GTVN -> CORRECT_BOTH
        # dont change drawing mode, will interrupt user correcting gtvn
        if self.cur_idl_step == IDLStep.CORRECT_BOTH:
            self._btn["next.step"].setEnabled(True)

        # (4-2) WAITING -> CORRECT_GTVT
        elif self.cur_idl_step == IDLStep.CORRECT_GTVT:
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.drawing_mode = DrawingMode.GTVT_PEN
            # change mouse cursor after:
            # (1) idl step updated
            # (2) drawing mode updated
            self.change_mouse_cursor(check_mouse_hover=True)

        # (5) end and start timer
        self.__timer[IDLStep.WAITING_GTVT].end()
        self.__timer[IDLStep.CORRECT_GTVT].start()

    def __confirm_gtvn_center(self):
        # (1) detect gtvn clicks
        if len(self.gtvn_clicks_pos_3d) == 0:
            text = (
                "No GTVn clicks detected. "
                "If current patient does not have GTVn, choose 'Yes' to continue. "
                "Choose 'No' to return and click GTVn."
            )
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
            # no gtvn click
            else:
                self.__timer[IDLStep.CLICK_GTVN_CENTER].end()
                self.__timer[IDLStep.WAITING_GTVN].start()
                self.__on_idl_gtvn_thread_finished()
                return

        # (2) update status
        if self.__idl_gtvt_thread.is_running:
            self.__update_cur_idl_step(IDLStep.WAITING)
        else:
            self.__update_cur_idl_step(IDLStep.CORRECT_GTVT)

        # (3) add clicks into 3d img
        if self.img_3d["gtvn.clicks"] is None:
            self.img_3d["gtvn.clicks"] = np.zeros_like(self.img_3d[Modal.CT])
        for pos in self.gtvn_clicks_pos_3d:
            # pos 0-transverse 1-coronal 2-saggital
            self.img_3d["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1

        # (4) transform gtvn clicks ndarray for idl.gtvn thread
        # copy data (dont change origin ndarray)
        idl_gtvn_clicks = self.img_3d["gtvn.clicks"].copy()
        # no need to flip empty img
        if idl_gtvn_clicks.max() > 0:
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=2)
            # turn upside down
            idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=0)

        # (5) start idl gtvn thread
        self.__idl_gtvn_thread.set_param(
            idl_gtvn_id=self._idl_id["gtvn"],
            patient=self._cur_patient,
            idl_gtvn_clicks=idl_gtvn_clicks,
            dataset_part=self._dataset_part,
            dataset_ver=self._dataset_ver,
            debug_mode=self._debug_mode,
        )
        self.__idl_gtvn_thread.start()

        # (6) refresh todolist and imgs, delete crosses
        self.__refresh_todo_list()
        # refresh contours only (on all img frames)
        self.refresh_imgs(
            reload_origin_rgb=False,
            reload_zoomed_rgb=False,
        )
        self.delete_all_crosses()

        # (7) update widgets
        if self.cur_idl_step == IDLStep.CORRECT_GTVT:
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.drawing_mode = DrawingMode.GTVT_PEN
            self.__enable_annotation_tools()
        elif self.cur_idl_step == IDLStep.WAITING:
            self.__disable_annotation_tools()
        # temporarily disable "next.step" button, until idl_step=CORRECT_BOTH
        self._btn["next.step"].setEnabled(False)

        # (8) start and end timer
        self.__timer[IDLStep.CLICK_GTVN_CENTER].end()
        self.__timer[IDLStep.WAITING_GTVN].start()

    def __update_idl_gtvn_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        Value.limit_range(progress_int, (0, 100))
        self.__progress_bar["gtvn"].setValue(progress_int)

    def __on_idl_gtvn_thread_finished(self):
        # (1) update status
        if self.__idl_gtvt_thread.is_running:
            self.__update_cur_idl_step(IDLStep.CORRECT_GTVN)
        else:
            self.__update_cur_idl_step(IDLStep.CORRECT_BOTH)

        # (2) load and combine 3d imgs
        self._load_idl_gtvn_data()
        self.__combine_pred_delineation_correction()
        # init correction and mask
        # (they are empty anyway, its efficient to init them after __combine_pred_delineation_correction)
        for i in ["gtvn.correction", "gtvn.correction.mask"]:
            self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])

        # (3) refresh todolist and imgs
        self.__refresh_todo_list()
        # refresh contours only (on all img frames)
        self.refresh_imgs(
            reload_origin_rgb=False,
            reload_zoomed_rgb=False,
        )

        # (4) update widgets
        self.__enable_annotation_tools()
        # (4-1) CORRECT_GTVT -> CORRECT_BOTH
        # dont change drawing mode, will interrupt user correcting gtvt
        if self.cur_idl_step == IDLStep.CORRECT_BOTH:
            self._btn["next.step"].setEnabled(True)

        # (4-2) WAITING -> CORRECT_GTVN
        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            self._radio_btn["correct.gtvn"].setChecked(True)
            self.drawing_mode = DrawingMode.GTVN_PEN
            # change mouse cursor after:
            # (1) idl step updated
            # (2) drawing mode updated
            self.change_mouse_cursor(check_mouse_hover=True)

        # (5) end and start timer
        self.__timer[IDLStep.WAITING_GTVN].end()
        self.__timer[IDLStep.CORRECT_GTVN].start()

    # this function is connected to widget, dont set input params to this function
    def __on_btn_next_step_clicked(self):
        if self.cur_idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.__confirm_gtvt_center()
        elif self.cur_idl_step == IDLStep.DRAW_GTVT:
            self.__confirm_gtvt_delineation()
        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.__confirm_gtvn_center()
        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            self.__save_corrections_and_masks()
            self.__goto_idl_step_approved()

    # check delineation in 3 different planes
    def __update_gtvt_delineated_status(self) -> Dict:
        if self.img_3d["gtvt.delineation"] is None:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                self.__gtvt_delineated_state[plane] = False
                self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()

        else:
            d, h, w = np.where(self.img_3d["gtvt.click"] == 1)
            d, h, w = int(d), int(h), int(w)
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                if plane == Plane.TRANSVERSE:
                    cur_plane_delineation = self.img_3d["gtvt.delineation"][
                        d, :, :
                    ].copy()
                    cur_plane_delineation[h, :] = 0
                    cur_plane_delineation[:, w] = 0

                elif plane == Plane.CORONAL:
                    cur_plane_delineation = self.img_3d["gtvt.delineation"][
                        :, h, :
                    ].copy()
                    cur_plane_delineation[d, :] = 0
                    cur_plane_delineation[:, w] = 0

                elif plane == Plane.SAGITTAL:
                    cur_plane_delineation = self.img_3d["gtvt.delineation"][
                        :, :, w
                    ].copy()
                    cur_plane_delineation[d, :] = 0
                    cur_plane_delineation[:, h] = 0

                if cur_plane_delineation.max() == 0:
                    self.__gtvt_delineated_state[plane] = False
                    self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()
                else:
                    self.__gtvt_delineated_state[plane] = True
                    self._text_label["draw.gtvt.{}".format(plane)].set_status_done()

    def __change_color(self, pixmap: QtGui.QPixmap, old_color, new_color):
        image = pixmap.toImage()
        old_qcolor = QtGui.QColor(*old_color)  # Unpack the tuple
        new_qcolor = QtGui.QColor(*new_color)  # Unpack the tuple

        for x in range(image.width()):
            for y in range(image.height()):
                if image.pixelColor(x, y) == old_qcolor:
                    image.setPixelColor(x, y, new_qcolor)
        return QtGui.QPixmap.fromImage(image)

    def restore_mouse_cursor(self):
        self.setCursor(Qt.ArrowCursor)

    def change_mouse_cursor(
        self,
        check_mouse_hover: bool = False,  # if = True, only change cursor when mouse is on a img
    ):
        if self.cur_idl_step not in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            return

        if check_mouse_hover:
            is_mouse_over_img = False
            for i in self.img_frame.keys():
                if self.img_frame[i].underMouse():
                    is_mouse_over_img = True
            if not is_mouse_over_img:
                return

        # set cursor center based on cursor size
        cursor_size = 32
        if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
            tool = "pen"
            left = 0
            top = cursor_size * 0.95
        elif self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            tool = "eraser"
            left = cursor_size * 0.2
            top = cursor_size * 0.8
        elif self.drawing_mode in [DrawingMode.GTVT_CLEAR, DrawingMode.GTVN_CLEAR]:
            tool = "clear"
            left = cursor_size * 0.5
            top = cursor_size * 0.5
        elif self.drawing_mode in [DrawingMode.GTVT_RESTORE, DrawingMode.GTVN_RESTORE]:
            tool = "restore"
            left = cursor_size * 0.5
            top = cursor_size * 0.5

        # CORRECT_BOTH is not a key of self.__cursor
        # change into CORRECT_GTVT or CORRECT_GTVN based on drawing mode
        if self.cur_idl_step == IDLStep.CORRECT_BOTH:
            if self.drawing_mode in [
                DrawingMode.GTVT_PEN,
                DrawingMode.GTVT_ERASER,
                DrawingMode.GTVT_CLEAR,
                DrawingMode.GTVT_RESTORE,
            ]:
                idl_step = IDLStep.CORRECT_GTVT
            elif self.drawing_mode in [
                DrawingMode.GTVN_PEN,
                DrawingMode.GTVN_ERASER,
                DrawingMode.GTVN_CLEAR,
                DrawingMode.GTVN_RESTORE,
            ]:
                idl_step = IDLStep.CORRECT_GTVN

        else:  # DRAW_GTVT / CORRECT_GTVT / CORRECT_GTVN
            idl_step = self.cur_idl_step

        cursor_pixmap = self.__cursor[idl_step][tool]
        self.setCursor(QtGui.QCursor(cursor_pixmap, left, top))

    def _init_widgets_cursor(self):
        self.__cursor = Dict()
        cursor_size = 32  # cursor size is no larger than 32
        origin_color = (0, 0, 0)
        for tool in ["pen", "eraser", "clear", "restore"]:
            origin_cursor = QtGui.QPixmap(
                (os.path.join(g.PROJ_DIR, "icons", "{}_cursor.png".format(tool)))
            )
            for idl_step in [
                IDLStep.DRAW_GTVT,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]:
                self.__cursor[idl_step][tool] = origin_cursor.scaled(
                    cursor_size,
                    cursor_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                # change color (after cursor pixmap is scaled to 32*32)
                # as __change_color is not efficiency
                if idl_step == IDLStep.DRAW_GTVT:
                    new_color = self.color["gtvt.delineation"]
                elif idl_step == IDLStep.CORRECT_GTVT:
                    new_color = self.color["gtvt.pred"]
                elif idl_step == IDLStep.CORRECT_GTVN:
                    new_color = self.color["gtvn.pred"]
                self.__cursor[idl_step][tool] = self.__change_color(
                    pixmap=self.__cursor[idl_step][tool],
                    old_color=origin_color,
                    new_color=new_color,
                )

    def _init_color(self, ui_setting: Dict):
        super()._init_color(ui_setting)
        self.color["eraser"] = self.color["black"]  # transparent
        self.color["gtvt.correction"] = self.color["gtvt.pred"]
        self.color["gtvn.correction"] = self.color["gtvn.pred"]
        self.color["gtvt.pred.final"] = self.color["gtvt.pred"]
        self.color["gtvn.pred.final"] = self.color["gtvn.pred"]

        # colors for idl mode only
        for i in ["gtvt.click", "gtvn.clicks", "gtvt.delineation"]:
            self.color[i] = self.color[ui_setting["color.contour"]["{}.idl".format(i)]]

    # this function is connected to widget, dont set input params to this function
    def __on_btn_restore_clicked(self):
        if self.cur_idl_step not in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            return

        # update drawing mode
        if self.cur_idl_step == IDLStep.CORRECT_GTVT:
            self.drawing_mode = DrawingMode.GTVT_RESTORE

        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            self.drawing_mode = DrawingMode.GTVN_RESTORE

        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_RESTORE
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_RESTORE

    # this function is connected to widget, dont set input params to this function
    def __on_btn_clear_clicked(self):
        if self.cur_idl_step not in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            return

        # update drawing mode
        if self.cur_idl_step in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
        ]:
            self.drawing_mode = DrawingMode.GTVT_CLEAR

        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            self.drawing_mode = DrawingMode.GTVN_CLEAR

        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_CLEAR
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_CLEAR

    def __get_gtvt_center_slices_id(self):
        if self.gtvt_click_pos_3d is None:
            Debug.error_exit("self.gtvt_click_pos_3d is empty")
        else:
            center_slices_id = Dict()
            center_slices_id[Plane.TRANSVERSE] = self.gtvt_click_pos_3d[0]
            center_slices_id[Plane.CORONAL] = self.gtvt_click_pos_3d[1]
            center_slices_id[Plane.SAGITTAL] = self.gtvt_click_pos_3d[2]
        return center_slices_id

    def __get_gtvn_center_slices_id(self):
        if len(self.gtvn_clicks_pos_3d) == 0:
            Debug.error_exit("self.gtvn_clicks_pos_3d is empty")
        else:
            center_slices_id = Dict()
            center_slices_id[Plane.TRANSVERSE] = self.gtvn_clicks_pos_3d[-1][0]
            center_slices_id[Plane.CORONAL] = self.gtvn_clicks_pos_3d[-1][1]
            center_slices_id[Plane.SAGITTAL] = self.gtvn_clicks_pos_3d[-1][2]
        return center_slices_id

    def delete_all_crosses(self):
        for i in [
            Modal.CT,
            Modal.PT,
            Modal.MR1,
            Modal.MR2,
            Plane.TRANSVERSE,
            Plane.CORONAL,
            Plane.SAGITTAL,
        ]:
            self.img_frame[i].delete_all_crosses()

    def refresh_crosses(self, frame_name: str = None):
        if self.cur_idl_step not in [
            IDLStep.CLICK_GTVT_CENTER,
            IDLStep.CLICK_GTVN_CENTER,
        ]:
            return

        if frame_name is not None:
            frame_name_list = [frame_name]
        else:
            if self.display_mode() == DisplayMode.PLANE_FIXED:
                frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
            else:
                frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]

        for i in frame_name_list:
            self.img_frame[i].refresh_crosses()

    def update_cross_id(
        self,
        cross: DragCross,
        old_cross_id: tuple,
        new_cross_id: tuple,
    ):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                cross = self.img_frame[i].get_cross_by_id(old_cross_id)
                cross.cross_id = new_cross_id
        else:
            cross.cross_id = new_cross_id

    def remove_3d_pos_of_selected_cross(self, cross: DragCross):

        if self.cur_idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.gtvt_click_pos_3d = None

        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            pos_3d = cross.cross_id
            if pos_3d in self.gtvn_clicks_pos_3d:
                self.gtvn_clicks_pos_3d.remove(pos_3d)

    def set_crosses_dragging_offset(self, img_frame: ImgFrame, pos: QPoint):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_frame[i].selected_cross.offset = pos
        else:
            img_frame.selected_cross.offset = pos

    def set_crosses_dragging_state(self, img_frame: ImgFrame, dragging: bool):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_frame[i].selected_cross.dragging = dragging
        else:
            img_frame.selected_cross.dragging = dragging

    def move_cross(self, img_frame: ImgFrame, pos: QPoint):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_frame[i].selected_cross.move(pos)
        else:
            img_frame.selected_cross.move(pos)

    def delete_selected_crosses(self):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]

        for i in frame_name_list:
            self.img_frame[i].delete_selected_cross()

    def select_cross(self, cross_id: tuple):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
        for i in frame_name_list:
            self.img_frame[i].select_cross(cross_id)

    def get_nii_spacing(self):
        return self._nii_spacing

    def get_3d_img_shape(self):
        if self.img_3d[Modal.CT] is not None:
            return self.img_3d[Modal.CT].shape
        else:
            return None

    def __enable_annotation_tools(self):
        # annotation buttons
        for i in ["pen", "eraser", "clear"]:
            self._btn[i].setEnabled(True)
        if self.cur_idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self._btn["restore"].setEnabled(True)
            # self._btn["restore"].show()
        else:
            # self._btn["restore"].hide()
            self._btn["restore"].setEnabled(False)

        # pen/eraser size slider bars
        if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
            self._text_label["eraser.size"].hide()
            self._slider["eraser.size"].hide()
            self._text_label["pen.size"].show()
            self._slider["pen.size"].show()
        elif self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            self._text_label["pen.size"].hide()
            self._slider["pen.size"].hide()
            self._text_label["eraser.size"].show()
            self._slider["eraser.size"].show()

        # radio buttons: correct gtvt/gtvn
        for i in ["correct.gtvt", "correct.gtvn"]:
            if self.cur_idl_step == IDLStep.CORRECT_BOTH:
                self._radio_btn[i].show()
            else:
                self._radio_btn[i].hide()

        self._collap["annotation"].expand()

    def __disable_annotation_tools(self):
        for i in ["pen", "eraser", "clear", "restore"]:
            self._btn[i].setEnabled(False)
        for i in ["pen.size", "eraser.size"]:
            self._text_label[i].hide()
            self._slider[i].hide()
        for i in ["correct.gtvt", "correct.gtvn"]:
            self._radio_btn[i].hide()

    def _init_widgets_annotation(self, ui_setting: Dict):
        # load pen/eraser size
        self.__pen_size_min = ui_setting["pen.size.min"]
        self.__pen_size_step = ui_setting["pen.size.step"]
        self.__eraser_size_min = ui_setting["eraser.size.min"]
        self.__eraser_size_step = ui_setting["eraser.size.step"]

        # text label
        for i in ["gtvt.progress", "gtvn.progress", "pen.size", "eraser.size"]:
            self._text_label[i] = QtWidgets.QLabel()
            self._text_label[i].setFixedHeight(g.TEXT_HEIGHT)
            self._text_label[i].hide()

        # set text
        self._text_label["pen.size"].setText("Pen Size")
        self._text_label["eraser.size"].setText("Eraser Size")
        self._text_label["gtvt.progress"].setText("Generating GTVt")
        self._text_label["gtvn.progress"].setText("Generating GTVn")

        # radio button
        for i in ["gtvt", "gtvn"]:
            self._radio_btn["correct.{}".format(i)] = QtWidgets.QRadioButton()
            self._radio_btn["correct.{}".format(i)].setFixedHeight(g.TEXT_HEIGHT)
        self._radio_btn["correct.gtvt"].setText("Correct GTVt")
        self._radio_btn["correct.gtvn"].setText("Correct GTVn")
        self._radio_btn["correct.gtvt"].setChecked(True)

        # annotation buttons
        for i in ["pen", "eraser", "clear", "restore"]:
            self._btn[i] = QtWidgets.QPushButton()
            height = 40 if g.is_linux() else 60
            self._btn[i].setFixedHeight(height)
            # add too tip and set stylesheet
            self._btn[i].setStyleSheet(
                """
                QToolTip {
                    font-weight: light;
                    color: dark-gray;
                    background-color: #fff;
                    border: 1px solid black;
                }
            """
            )
            self._btn[i].setToolTip(i.capitalize())
            # set btn icons
            icon = QtGui.QIcon(
                os.path.join(g.PROJ_DIR, "icons", "{}_btn.png".format(i))
            )
            if i == "pen":
                icon_size = 24 if g.is_linux() else 36
            elif i == "eraser":
                icon_size = 31 if g.is_linux() else 46
            elif i == "clear":
                icon_size = 33 if g.is_linux() else 48
            elif i == "restore":
                icon_size = 28 if g.is_linux() else 42
            self._btn[i].setIconSize(QSize(icon_size, icon_size))
            self._btn[i].setIcon(icon)
            self._btn[i].setEnabled(False)

        # connect btns to functions
        self._btn["pen"].clicked.connect(self.__on_btn_pen_clicked)
        self._btn["eraser"].clicked.connect(self.__on_btn_eraser_clicked)
        self._btn["clear"].clicked.connect(self.__on_btn_clear_clicked)
        self._btn["restore"].clicked.connect(self.__on_btn_restore_clicked)

        # gtvt/gtvn progress bars
        self.__progress_bar = Dict()
        for i in ["gtvt", "gtvn"]:
            self.__progress_bar[i] = QtWidgets.QProgressBar()
            self.__progress_bar[i].setFixedHeight(g.TEXT_HEIGHT)
            self.__progress_bar[i].setRange(0, 100)
            self.__progress_bar[i].setValue(0)
            self.__progress_bar[i].hide()

        # pen/eraser size slider
        for i in ["pen.size", "eraser.size"]:
            self._slider[i] = QtWidgets.QSlider()
            self._slider[i].setFixedHeight(g.SLIDER_HEIGHT)
            self._slider[i].setOrientation(Qt.Horizontal)
            self._slider[i].hide()
            self._slider[i].setMinimum(0)
            self._slider[i].setMaximum(2)
        self._slider["pen.size"].setValue(0)
        self._slider["eraser.size"].setValue(1)

        self.__radio_group_drawing_mode = QtWidgets.QButtonGroup()
        for i in ["gtvt", "gtvn"]:
            self.__radio_group_drawing_mode.addButton(
                self._radio_btn["correct.{}".format(i)]
            )
            self._radio_btn["correct.{}".format(i)].hide()
        self.__radio_group_drawing_mode.buttonClicked.connect(
            self.__switch_drawing_mode_gtv
        )

        # create qcollapsible space
        self._collap["annotation"] = QCollapsible("ANNOTATION TOOLS")
        self._collap["annotation"].expand()
        v_layout = QtWidgets.QVBoxLayout()

        # (1) add annotation buttons
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.setSpacing(20)
        for i in ["pen", "eraser", "clear", "restore"]:
            h_layout.addWidget(self._btn[i])
        v_layout.addLayout(h_layout)

        # (2) add radio buttons
        h_layout = QtWidgets.QHBoxLayout()
        for i in ["gtvt", "gtvn"]:
            h_layout.addWidget(self._radio_btn["correct.{}".format(i)])
        v_layout.addLayout(h_layout)

        # (3) add pen/eraser size slider
        for i in ["pen.size", "eraser.size"]:
            v_layout.addWidget(self._text_label[i])
            v_layout.addWidget(self._slider[i])

        # (4) add progress bars and labels
        for i in ["gtvt", "gtvn"]:
            v_layout.addWidget(self._text_label["{}.progress".format(i)])
            v_layout.addWidget(self.__progress_bar[i])

        # idl gtvt/gtvn thread (after progress bars and progress bar labels initialized)
        self.__idl_gtvt_thread = IDLGTVtThread(
            progress_bar=self.__progress_bar["gtvt"],
            progress_bar_label=self._text_label["gtvt.progress"],
        )
        self.__idl_gtvt_thread.progress_signal.connect(
            self.__update_idl_gtvt_progress_bar
        )
        self.__idl_gtvt_thread.complete_signal.connect(
            self.__on_idl_gtvt_thread_finished
        )
        self.__idl_gtvn_thread = IDLGTVnThread(
            progress_bar=self.__progress_bar["gtvn"],
            progress_bar_label=self._text_label["gtvn.progress"],
        )
        self.__idl_gtvn_thread.progress_signal.connect(
            self.__update_idl_gtvn_progress_bar
        )
        self.__idl_gtvn_thread.complete_signal.connect(
            self.__on_idl_gtvn_thread_finished
        )

        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)
        self._collap["annotation"].addWidget(container)

    def _init_widgets(self, ui_setting: Dict):
        super()._init_widgets(ui_setting)

        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._collap[i].collapse()
            self._collap[i].hide()

        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].collapse()

    def _clear_img_3d(self):
        super()._clear_img_3d()
        for i in ["gtvt.correction.mask", "gtvn.correction.mask"]:
            self.img_3d[i] = None

    def _init_data(self, ui_setting: Dict):
        super()._init_data(ui_setting)

        # init baseline id and idl.gtvt/gtvn id, keep them unchanged
        self._baseline_id = "baseline_real.idl"

        # (1) new training
        # initlize idl.gtvt/gtvn id
        if self.__train_id == "Start a new experiment":
            cur_time = Timer.cur_time_str()
            for i in ["gtvt", "gtvn"]:
                self._idl_id[i] = "idl.{}_".format(i) + cur_time

                if self.__user_name != "" and self.__user_name is not None:
                    while self.__user_name.startswith("_"):
                        self.__user_name = self.__user_name[1:]
                    while self.__user_name.endswith("_"):
                        self.__user_name = self.__user_name[:-1]
                    self._idl_id[i] += "_" + self.__user_name

                if self._debug_mode:
                    self._idl_id[i] += "_" + Debug.DELETE_FLAG

            # create idl.gtvt/gtvn folders
            for i in ["gtvt", "gtvn"]:
                Dir.create(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i]
                    )
                )

        # (2) existing train id
        else:
            for i in ["gtvt", "gtvn"]:
                self._idl_id[i] = "idl.{}_{}".format(i, self.__train_id)

        # initialize the position of gtvt click / gtvn clicks
        self.gtvt_click_pos_3d = None
        self.gtvn_clicks_pos_3d = List()

        # drawing
        self.drawing_mode = DrawingMode.GTVT_PEN
        self.paint_pos = None  # Store the last painted point
        self.__gtvt_delineated_state = Dict()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.__gtvt_delineated_state[plane] = False

        # init idl_step and json file
        self.cur_idl_step = None
        self.__idl_step_json_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id["gtvt"],  # only save it in gtvt folder
            "idl_step.json",
        )
        if not os.path.exists(self.__idl_step_json_path):
            Json.save({}, self.__idl_step_json_path)

        # load/save interpolation step
        interpolation_setting_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id["gtvt"],
            "interpolation.json",
        )
        if os.path.exists(interpolation_setting_path):
            self.interpolation_step = Json.load(interpolation_setting_path)["step"]
            self.interpolation_step = max(1, int(self.interpolation_step))
        else:
            self.interpolation_step = ui_setting["interpolation.step"]
            self.interpolation_step = max(1, int(self.interpolation_step))
            Json.save({"step": self.interpolation_step}, interpolation_setting_path)

    def __save_idl_step(self):
        idl_step_of_all_patients = Json.load(self.__idl_step_json_path)
        idl_step_of_all_patients["patient={}".format(self._cur_patient)] = (
            self.cur_idl_step
        )
        Json.save(idl_step_of_all_patients, self.__idl_step_json_path)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_F12:
            pass

        # delete selected cross
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected_crosses()

        super().keyPressEvent(event)

    def __clear_all_drawing_layers(self, img_frame: ImgFrame):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            frame_name_list = [img_frame.plane]
        for i in frame_name_list:
            self.img_frame[i].drawing_layer = QtGui.QPixmap(self.img_frame[i].size())
            self.img_frame[i].drawing_layer.fill(Qt.transparent)
            self.img_frame[i].update()

    # this function is connected to widget, dont set input params to this function
    def __switch_drawing_mode_gtv(self):
        # gtvn to gtvt
        if self._radio_btn["correct.gtvt"].isChecked():
            # pen
            if self.drawing_mode == DrawingMode.GTVN_PEN:
                self.drawing_mode = DrawingMode.GTVT_PEN
            # eraser
            elif self.drawing_mode == DrawingMode.GTVN_ERASER:
                self.drawing_mode = DrawingMode.GTVT_ERASER
            # clear
            elif self.drawing_mode == DrawingMode.GTVN_CLEAR:
                self.drawing_mode = DrawingMode.GTVT_CLEAR
            # restore
            elif self.drawing_mode == DrawingMode.GTVN_RESTORE:
                self.drawing_mode = DrawingMode.GTVT_RESTORE

        # gtvt to gtvn
        elif self._radio_btn["correct.gtvn"].isChecked():
            # pen
            if self.drawing_mode == DrawingMode.GTVT_PEN:
                self.drawing_mode = DrawingMode.GTVN_PEN
            # eraser
            elif self.drawing_mode == DrawingMode.GTVT_ERASER:
                self.drawing_mode = DrawingMode.GTVN_ERASER
            # clear
            elif self.drawing_mode == DrawingMode.GTVT_CLEAR:
                self.drawing_mode = DrawingMode.GTVN_CLEAR
            # restore
            elif self.drawing_mode == DrawingMode.GTVT_RESTORE:
                self.drawing_mode = DrawingMode.GTVN_RESTORE

    def get_pen_size(self):
        pen_size = (
            self._slider["pen.size"].value() * self.__pen_size_step
            + self.__pen_size_min
        )
        pen_size *= self.get_zoomin_factor()
        return pen_size

    def get_eraser_size(self):
        eraser_size = (
            self._slider["eraser.size"].value() * self.__eraser_size_step
            + self.__eraser_size_min
        )
        eraser_size *= self.get_zoomin_factor()
        return eraser_size

    def _load_baseline_data(self):
        # self._reset_zoomin()
        self._clear_img_3d()
        self._clear_img_frames()

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self.combox["patient"].setCurrentIndex(-1)  # show nothing

    def _add_instruction_on_top_left(self, qimg: QtGui.QImage):
        left = self._get_text_pos_left()[0]
        top = self._get_text_pos_top()

        if self._cur_patient is None:
            text = "Please select a patient"
            self._qimg_draw_text(
                qimg=qimg,
                text=text,
                pos=(left, top),
                color=self.color["green"],
            )
            return

        if self.cur_idl_step == IDLStep.CLICK_GTVT_CENTER:
            text = "Please click the center of primary Gross Tumor Volumes (GTVt)"
        elif self.cur_idl_step == IDLStep.DRAW_GTVT:
            text = "Please delineate GTVt in 3 anatomical planes"
        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            text = "Please click the center of malignant lymph nodes (GTVn)"
        elif self.cur_idl_step == IDLStep.WAITING:
            text = "Neural Network is generating auto-segmentation, please wait..."
        elif self.cur_idl_step == IDLStep.CORRECT_GTVT:
            text = "Please correct the GTVt auto-segmentation"
        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            text = "Please correct the GTVn auto-segmentation"
        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            text = "Please correct the GTVt and GTVn auto-segmentations"
        elif self.cur_idl_step == IDLStep.APPROVED:
            text = "Auto-segmentations of current patient are finalized"

        self._qimg_draw_text(
            qimg=qimg,
            text=text,
            pos=(left, top),
            color=self.color["green"],
        )

        # show delineated state on qimage
        if self.cur_idl_step == IDLStep.DRAW_GTVT:
            top += 5
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                top += 20
                text = plane.capitalize()
                if self.__gtvt_delineated_state[plane] is True:
                    text += " ✓"
                    color = self.color["green"]
                else:
                    text += " ✕"
                    color = self.color["red"]
                self._qimg_draw_text(
                    qimg=qimg,
                    text=text,
                    pos=(left, top),
                    color=color,
                )

    # rewrite this function (do nothing)
    def _add_score_on_top_left(self, qimg: QtGui.QImage):
        pass

    def _add_contour_description_on_bottom_left(self, qimg: QtGui.QImage):
        left = self._get_text_pos_left()
        bottom = self._get_text_pos_bottom(qimg)

        # user input text
        if (
            self.img_3d["gtvt.click"] is not None
            or self.img_3d["gtvt.delineation"] is not None
            or self.img_3d["gtvn.clicks"] is not None
        ):
            self._qimg_draw_text(
                qimg=qimg,
                text="User Input",
                pos=(left[0], bottom[0]),
                color=self.color["gtvt.delineation"],
            )

        # gtvt pred text
        if self.img_3d["gtvt.pred"] is not None:
            self._qimg_draw_text(
                qimg=qimg,
                text="GTVt - Pred",
                pos=(left[0], bottom[1]),
                color=self.color["gtvt.pred"],
            )

        # gtvn pred text
        if self.img_3d["gtvn.pred"] is not None:
            if self.img_3d["gtvt.pred"] is None:
                pos = (left[0], bottom[1])
            else:
                pos = (left[0], bottom[2])
            self._qimg_draw_text(
                qimg=qimg,
                text="GTVn - Pred",
                pos=pos,
                color=self.color["gtvn.pred"],
            )

    def _load_patient_data(self):
        # stop idl qthreads (if running)
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # update widgets
        self._collap["patient"].collapse()
        self._collap["annotation"].expand()
        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].setEnabled(True)

        # clear data
        self._clear_img_3d()
        self.gtvt_click_pos_3d = None
        self.gtvn_clicks_pos_3d = List()

        # update current patient
        self._cur_patient = self.combox["patient"].currentText()

        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_dir_and_nii_spacing()

        # load multi-modal imgs only, no labels
        self._load_multi_modal_imgs()

        # load idl gtvt/gtvn images
        self._load_idl_gtvt_data()
        self._load_idl_gtvn_data()

        # reset timers after cur_patient is updated
        self.__timer = Dict()
        for i in [
            IDLStep.CLICK_GTVT_CENTER,
            IDLStep.DRAW_GTVT,
            IDLStep.CLICK_GTVN_CENTER,
            IDLStep.WAITING_GTVT,
            IDLStep.WAITING_GTVN,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
        ]:
            self.__timer[i] = IDLTimer(
                baseline_id=self._baseline_id,
                idl_gtvt_id=self._idl_id["gtvt"],
                patient=self._cur_patient,
                idl_step=i,
            )

        # load current idl step from json(after cur_patient is updated)
        # init and save idl step of all patients
        idl_step_of_all_patients = Json.load(self.__idl_step_json_path)
        self.cur_idl_step = idl_step_of_all_patients[
            "patient={}".format(self._cur_patient)
        ]

        # adjust cur_idl_step
        # (1) new patient
        if self.cur_idl_step == {}:
            self.cur_idl_step = IDLStep.CLICK_GTVT_CENTER

        # (2) cur_idl_step == WAITING or CORRECT_GTVN means gtvt thread was interupted
        elif self.cur_idl_step in [IDLStep.WAITING, IDLStep.CORRECT_GTVN]:
            # goto the nearest recoverable step
            self.cur_idl_step = IDLStep.DRAW_GTVT

        # (3) cur_idl_step == CORRECT_GTVT means gtvn thread was interupted
        elif self.cur_idl_step == IDLStep.CORRECT_GTVT:
            # goto the nearest recoverable step
            self.cur_idl_step = IDLStep.CLICK_GTVN_CENTER

        # (4) cur_idl_step == CLICK_GTVN_CENTER means gtvt thread might be interupted
        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            # no gtvt pred menas gtvt thread was interupted
            if self.img_3d["gtvt.pred"] is None:
                # goto the nearest recoverable step
                self.cur_idl_step = IDLStep.DRAW_GTVT

        # call reset_cur_slice_id() after:
        # (1) _load_multi_modal_imgs
        # (2) _load_idl_gtvt_data(), will load gtvt_click_pos_3d
        # (3) _load_idl_gtvn_data(), will load gtvn_clicks_pos_3d
        # (4) self.cur_idl_step is loaded
        self.reset_cur_slice_id()

        # last step: goto current idl step
        if self.cur_idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.__goto_idl_step_click_gtvt_center()

        elif self.cur_idl_step == IDLStep.DRAW_GTVT:
            self.__goto_idl_step_draw_gtvt()

        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.__goto_idl_step_click_gtvn_center()

        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            self.__goto_idl_step_correct_both()

        elif self.cur_idl_step == IDLStep.APPROVED:
            self.__goto_idl_step_approved()

    def ensure_slice_id_multiple(self, slice_id: int, slice_count: int):
        remainder = slice_id % self.interpolation_step
        slice_id -= remainder
        if (
            remainder > self.interpolation_step / 2
            and slice_id + self.interpolation_step <= slice_count - 1
        ):
            slice_id += self.interpolation_step
        slice_id = Value.limit_range(slice_id, (0, slice_count - 1))
        return slice_id

    def reset_cur_slice_id(self):
        if (
            self.cur_idl_step == IDLStep.CLICK_GTVT_CENTER
            or self.cur_idl_step == IDLStep.DRAW_GTVT
        ):
            if self.gtvt_click_pos_3d is None:
                self.cur_slice_id[Plane.CORONAL] = self.img_3d[Modal.CT].shape[1] // 2
                self.cur_slice_id[Plane.SAGITTAL] = self.img_3d[Modal.CT].shape[2] // 2
                self.cur_slice_id[Plane.TRANSVERSE] = (
                    self.img_3d[Modal.CT].shape[0] // 2
                )
                # make sure transverse slice id is a multiple of interpolation step
                self.cur_slice_id[Plane.TRANSVERSE] = self.ensure_slice_id_multiple(
                    slice_id=self.cur_slice_id[Plane.TRANSVERSE],
                    slice_count=self.img_3d[Modal.CT].shape[0],
                )

            else:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()

        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            if len(self.gtvn_clicks_pos_3d) == 0:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()
            else:
                self.cur_slice_id = self.__get_gtvn_center_slices_id()

        elif self.cur_idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()

            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                # sometimes there is not gtvn clicks, but there is always a gtvt click
                if len(self.gtvn_clicks_pos_3d) == 0:
                    self.cur_slice_id = self.__get_gtvt_center_slices_id()
                else:
                    self.cur_slice_id = self.__get_gtvn_center_slices_id()

        elif self.cur_idl_step == IDLStep.APPROVED:
            self.cur_slice_id = self.__get_gtvt_center_slices_id()

    def __update_cur_idl_step(self, new_idl_step: str):
        self.cur_idl_step = new_idl_step
        self.__save_idl_step()

    def __refresh_todo_list(self):
        if (
            self.cur_idl_step != IDLStep.CORRECT_BOTH
            and self.cur_idl_step != IDLStep.APPROVED
        ):
            self._text_label[self.cur_idl_step].set_status_ongoing()

        if self.cur_idl_step == IDLStep.CLICK_GTVT_CENTER:
            done_step_list = [IDLStep.SELECT_PATIENT]
            notstart_step_list = [
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]

        elif self.cur_idl_step == IDLStep.DRAW_GTVT:
            done_step_list = [IDLStep.SELECT_PATIENT, IDLStep.CLICK_GTVT_CENTER]
            notstart_step_list = [
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]
            # sub steps of draw.gtvt
            self.__update_gtvt_delineated_status()

        elif self.cur_idl_step == IDLStep.CLICK_GTVN_CENTER:
            done_step_list = [
                IDLStep.SELECT_PATIENT,
                IDLStep.CLICK_GTVT_CENTER,
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
            ]
            notstart_step_list = [
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]
            if self.__idl_gtvt_thread.is_running:
                self._text_label[IDLStep.WAITING].set_status_ongoing()
            else:
                self._text_label[IDLStep.WAITING].set_status_notstart()

        elif self.cur_idl_step == IDLStep.WAITING:
            done_step_list = [
                IDLStep.SELECT_PATIENT,
                IDLStep.CLICK_GTVT_CENTER,
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
                IDLStep.CLICK_GTVN_CENTER,
            ]
            notstart_step_list = [
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]
        elif self.cur_idl_step == IDLStep.CORRECT_GTVT:
            done_step_list = [
                IDLStep.SELECT_PATIENT,
                IDLStep.CLICK_GTVT_CENTER,
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
                IDLStep.CLICK_GTVN_CENTER,
            ]
            notstart_step_list = [IDLStep.CORRECT_GTVN]
            self._text_label[IDLStep.WAITING].set_status_ongoing()

        elif self.cur_idl_step == IDLStep.CORRECT_GTVN:
            done_step_list = [
                IDLStep.SELECT_PATIENT,
                IDLStep.CLICK_GTVT_CENTER,
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
                IDLStep.CLICK_GTVN_CENTER,
            ]
            notstart_step_list = [IDLStep.CORRECT_GTVT]
            self._text_label[IDLStep.WAITING].set_status_ongoing()

        elif self.cur_idl_step == IDLStep.CORRECT_BOTH:
            self._text_label[IDLStep.CORRECT_GTVT].set_status_ongoing()
            self._text_label[IDLStep.CORRECT_GTVN].set_status_ongoing()
            done_step_list = [
                IDLStep.SELECT_PATIENT,
                IDLStep.CLICK_GTVT_CENTER,
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
            ]
            notstart_step_list = []

        elif self.cur_idl_step == IDLStep.APPROVED:
            done_step_list = [
                IDLStep.SELECT_PATIENT,
                IDLStep.CLICK_GTVT_CENTER,
                IDLStep.DRAW_GTVT,
                IDLStep.DRAW_GTVT_TRANSVERSE,
                IDLStep.DRAW_GTVT_CORONAL,
                IDLStep.DRAW_GTVT_SAGITTAL,
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]
            notstart_step_list = []

        for i in done_step_list:
            self._text_label[i].set_status_done()
        for i in notstart_step_list:
            self._text_label[i].set_status_notstart()

    def _load_idl_gtvt_data(self):
        self._load_idl_gtv_data(gtv="gtvt")
        # load gtvt click pos from 3d img
        if self.img_3d["gtvt.click"] is not None:
            pos = np.where(self.img_3d["gtvt.click"] == 1)
            self.gtvt_click_pos_3d = pos[0][0], pos[1][0], pos[2][0]

    def _load_idl_gtvn_data(self):
        self._load_idl_gtv_data(gtv="gtvn")
        if self.img_3d["gtvn.clicks"] is not None:
            pos = np.where(self.img_3d["gtvn.clicks"] == 1)
            self.gtvn_clicks_pos_3d = List(zip(*pos))

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
            nii_name_list += ["click", "delineation"]
        elif gtv == "gtvn":
            nii_name_list.append("clicks")

        for i in nii_name_list:
            nii_path = os.path.join(
                round_dir,
                "{}_{}.nii.gz".format(
                    gtv,
                    # "correction.mask" -> "correction_mask"
                    i.replace(".", "_"),
                ),
            )
            if os.path.exists(nii_path):
                self.img_3d["{}.{}".format(gtv, i)] = self._load_3d_img(
                    path=nii_path, binary=True
                )
            else:
                self.img_3d["{}.{}".format(gtv, i)] = None

    def __combine_pred_delineation_correction(self):
        if self.img_3d[Modal.CT] is None:
            return

        if self.cur_idl_step not in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            return

        for i in ["gtvt", "gtvn"]:
            # no pred loaded, generate an empty pred.final
            if self.img_3d["{}.pred".format(i)] is None:
                self.img_3d["{}.pred.final".format(i)] = np.zeros_like(
                    self.img_3d[Modal.CT]
                )
            # copy from origin pred
            else:
                self.img_3d["{}.pred.final".format(i)] = self.img_3d[
                    "{}.pred".format(i)
                ].copy()

            # # combine gtvt.pred and gtvt.delineation
            # if i == "gtvt":
            #     d, h, w = np.where(self.img_3d["gtvt.click"] == 1)
            #     self.img_3d["gtvt.pred.final"][d, :, :] = 0
            #     self.img_3d["gtvt.pred.final"][:, h, :] = 0
            #     self.img_3d["gtvt.pred.final"][:, :, w] = 0
            #     self.img_3d["gtvt.pred.final"] = np.maximum(
            #         self.img_3d["gtvt.pred.final"], self.img_3d["gtvt.delineation"]
            #     )

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
