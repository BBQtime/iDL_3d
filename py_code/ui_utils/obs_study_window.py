import math
import os
from multiprocessing import Process, Queue

import cv2
import global_utils.global_core as g
import numpy as np
import qimage2ndarray
import torch.multiprocessing as mp
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import (
    DisplayMode,
    DrawingMode,
    ErrMsg,
    Modal,
    ObsStudyGTVnStep,
    ObsStudyGTVtStep,
    Plane,
)
from numpy import ndarray
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QEvent, QPoint, QSize, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QGridLayout, QMessageBox
from scipy import ndimage
from superqt import QCollapsible
from training_utils.idl_gtvt_training import IDLGTVtTraining
from ui_utils.drag_cross import DragCross
from ui_utils.img_frame import ImgFrame
from ui_utils.interpolation import interpolate_shapes
from ui_utils.obs_study_thread import ObsStudyGTVnThread, ObsStudyGTVtProgressThread
from ui_utils.obs_study_timer import ObsStudyTimer
from ui_utils.replay_window import ReplayWindow
from ui_utils.todo_list_label import TodoListLabel


class ObsStudyWindow(ReplayWindow):
    def __init__(
        self,
        user_name: str,
        train_id: str,
    ):
        self.__user_name = user_name
        self.__train_id = train_id
        # pass debug_mode parameter to the parent class
        super().__init__()

        # Queue for communication
        if g.is_linux():
            # Set 'spawn' start method for linux system
            mp.set_start_method("spawn", force=True)
            self.__queue = mp.Queue()
        else:
            self.__queue = Queue()

    def draw_on_img_frame_press(self, event: QtGui.QMouseEvent, img_frame: ImgFrame):
        if (
            self.obs_study_gtvt_step
            not in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            return

        # get anatomical plane of current img frame
        plane = img_frame.plane

        # Switch to the GTVT center slice if the current slice is not the center
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            gtvt_center_slice_id = self.__get_gtvt_center_slices_id()[plane]
            if self.cur_slice_id[plane] != gtvt_center_slice_id:
                self.cur_slice_id[plane] = gtvt_center_slice_id

                # (1) PLANE_FIXED mode
                if self.display_mode() == DisplayMode.PLANE_FIXED:
                    # (1-1) refresh current img frame from scratch
                    cur_frame_name = img_frame.get_frame_name()
                    self.refresh_imgs(frame_name=cur_frame_name)
                    # (1-2) on other img frames, only refresh anatomical lines
                    frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                    frame_name_list.remove(cur_frame_name)
                    for i in frame_name_list:
                        self.refresh_imgs(
                            frame_name=i,
                            reload_origin_rgb=False,
                            reload_zoomed_rgb=False,
                            reload_contours=False,
                        )

                # (2) MODAL_FIXED mode, refresh all 4 img frames
                else:
                    self.refresh_imgs()

                # do nothing after switching to gtvt center slice
                return

        # (1) for pen/eraser mode, record paint position
        if self.drawing_mode in [
            DrawingMode.GTVT_PEN,
            DrawingMode.GTVN_PEN,
            DrawingMode.GTVT_ERASER,
            DrawingMode.GTVN_ERASER,
        ]:
            self.paint_pos = event.pos()
            # click (without moving) will also draw or erase
            self.draw_on_img_frame_move(event=event, img_frame=img_frame)
            return

        # (2) Clear the GTVT delineation on the current plane
        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE
            and self.drawing_mode
            in [
                DrawingMode.GTVT_CLEAR,
                DrawingMode.GTVN_CLEAR,
            ]
        ):
            # clear gtvt delineation of current plane
            self.img_3d["gtvt.delineation.{}".format(plane)] = np.zeros_like(
                self.img_3d[Modal.CT]
            )

            # update gtvt delineated state
            self.__gtvt_delineated_state[plane] = False

            # update todo list
            self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()

            # refresh contours only (on all img frames) after using "clear" tool
            self.refresh_imgs(
                reload_origin_rgb=False,
                reload_zoomed_rgb=False,
            )

        # (3) "clear" / "restore" in correction step
        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
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

            d = h = w = self.cur_slice_id[plane]
            correction = self.img_3d["{}.correction".format(gtv)]
            correction_mask = self.img_3d["{}.correction.mask".format(gtv)]

            if plane == Plane.TRANSVERSE:
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
                        for i in range(1, self.interpolation_step):
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

            elif plane == Plane.CORONAL:
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

            elif plane == Plane.SAGITTAL:
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

        if (
            self.obs_study_gtvt_step
            not in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            return

        # (1) set drawing color and size
        # (1-1) pen mode
        if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
            draw_size = self.get_pen_size()
            # delineate gtvt
            if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
                draw_color = QtGui.QColor(*self.color["gtvt.delineation"])
            # correct gtvt/gtvn
            elif (
                self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
                or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
            ):
                if self.drawing_mode in [
                    DrawingMode.GTVT_PEN,
                    DrawingMode.GTVT_ERASER,
                ]:
                    draw_color = QtGui.QColor(*self.color["gtvt.pred"])
                elif self.drawing_mode in [
                    DrawingMode.GTVN_PEN,
                    DrawingMode.GTVN_ERASER,
                ]:
                    draw_color = QtGui.QColor(*self.color["gtvn.pred"])
        # (1-2) eraser mode
        elif self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            draw_size = self.get_eraser_size()
            draw_color = QtGui.QColor(*self.color["eraser"])

        # (2) set pen
        pen = QtGui.QPen(draw_color)
        pen.setStyle(Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        # mouse didnt move, draw circle
        if self.paint_pos == event.pos():
            radius = draw_size / 4
            pen.setWidth(draw_size / 2)
        # mouse moved, draw line
        else:
            pen.setWidth(draw_size)

        # (3) loop through each image frame
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            frame_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            frame_name_list = [img_frame.plane]

        for i in frame_name_list:
            # create painter for current image frame
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
            # set pen for painter
            painter.setPen(pen)

            # set img frame's pen mode
            if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
                self.img_frame[i].pen_mode = False
            elif self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
                self.img_frame[i].pen_mode = True

            # mouse didnt move, draw circle
            if self.paint_pos == event.pos():
                painter.drawEllipse(event.pos(), radius, radius)
            # mouse moved, draw line
            else:
                painter.drawLine(self.paint_pos, event.pos())

            # repaint current image frame
            self.img_frame[i].update()

        # update paint pos in the last step
        self.paint_pos = event.pos()

    def draw_on_img_frame_release(self, img_frame: ImgFrame):
        if self.paint_pos is None:
            return

        if (
            self.obs_study_gtvt_step
            not in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            return

        plane = img_frame.plane
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
        new_drawing = g.binarize_img(img=new_drawing, threshold=binary_threshold)

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
        if plane == Plane.SAGITTAL:
            actual_shape = self.img_3d[Modal.CT][:, :, 0].shape
        elif plane == Plane.CORONAL:
            actual_shape = self.img_3d[Modal.CT][:, 0, :].shape
        elif plane == Plane.TRANSVERSE:
            actual_shape = self.img_3d[Modal.CT][0, :, :].shape
        new_drawing = cv2.resize(
            new_drawing,
            (actual_shape[1], actual_shape[0]),
            interpolation=cv2.INTER_AREA,  # best for scaling down
        )
        # binarization (after resize)
        new_drawing = g.binarize_img(img=new_drawing, threshold=binary_threshold)

        # get 2d existing drawing (gtvt delineation) from 3d ndarray
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            gtvt_delineation = self.img_3d["gtvt.delineation.{}".format(plane)]
            d, h, w = self.gtvt_click_pos_3d
            if plane == Plane.TRANSVERSE:
                exist_drawing = gtvt_delineation[d, :, :]
            elif plane == Plane.CORONAL:
                exist_drawing = gtvt_delineation[:, h, :]
            elif plane == Plane.SAGITTAL:
                exist_drawing = gtvt_delineation[:, :, w]

        # get 2d existing drawing (correction) from 3d ndarray
        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            d = h = w = self.cur_slice_id[plane]
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
            pred_final = self.img_3d["{}.pred.final".format(gtv)]
            # copy slice from 3d img, dont change the original 3d img
            if plane == Plane.TRANSVERSE:
                exist_drawing = pred_final[d, :, :].copy()
            elif plane == Plane.CORONAL:
                exist_drawing = pred_final[:, h, :].copy()
            elif plane == Plane.SAGITTAL:
                exist_drawing = pred_final[:, :, w].copy()

        # invert color if in eraser mode
        if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            exist_drawing = 1 - exist_drawing

        # combine exist_drawing and new_drawing
        new_drawing = np.maximum(exist_drawing, new_drawing)

        # fill holes if in pen mode
        if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
            new_drawing = ndimage.binary_fill_holes(new_drawing).astype(np.float32)

        # invert color back, if in eraser mode
        if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
            new_drawing = 1 - new_drawing

        # add 2d drawing into 3d gtvt delineation
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            gtvt_delineation = self.img_3d["gtvt.delineation.{}".format(plane)]
            # replace slice in 3d delineation
            if plane == Plane.TRANSVERSE:
                gtvt_delineation[d, :, :] = new_drawing
            elif plane == Plane.CORONAL:
                gtvt_delineation[:, h, :] = new_drawing
            elif plane == Plane.SAGITTAL:
                gtvt_delineation[:, :, w] = new_drawing

        # add 2d drawing into 3d correction
        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                correction = self.img_3d["gtvt.correction"]
                correction_mask = self.img_3d["gtvt.correction.mask"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                correction = self.img_3d["gtvn.correction"]
                correction_mask = self.img_3d["gtvn.correction.mask"]

            # replace slice in 3d correction
            if plane == Plane.TRANSVERSE:
                correction[d, :, :] = new_drawing
                correction_mask[d, :, :] = np.ones_like(new_drawing)
                self.__interpolation(
                    cur_slice_id=d,
                    correction=correction,
                    correction_mask=correction_mask,
                    pred_final=pred_final,
                )

            elif plane == Plane.CORONAL:
                correction[:, h, :] = new_drawing
                correction_mask[:, h, :] = np.ones_like(new_drawing)

            elif plane == Plane.SAGITTAL:
                correction[:, :, w] = new_drawing
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
            interpolated_slices = interpolate_shapes(
                start_slice=start_slice_data,
                end_slice=end_slice_data,
                steps=self.interpolation_step - 1,
            )

            interpolated_slices = g.binarize_img(interpolated_slices)

            # add interpolated slices
            correction[start_slice_id + 1 : end_slice_id, :, :] = interpolated_slices

            # update correction mask for interpolated slices
            for i in range(start_slice_id + 1, end_slice_id):
                correction_mask[i, :, :] = np.ones_like(correction[i, :, :])

    def __save_corrections_and_masks(self, gtv: str):
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
            # flip left/right
            img = np.flip(img, axis=2)
            # turn upside down
            img = np.flip(img, axis=0)
            # save
            g.save_nii(
                img=img,
                save_path=os.path.join(
                    cur_round_dir,
                    "{}_{}.nii.gz".format(
                        gtv,
                        # "correction.mask" -> "correction_mask"
                        i.replace(".", "_"),
                    ),
                ),
                spacing=g.NII_SPACING,
            )

    # this function is connected to widget, dont set input params to this function
    def __on_btn_pen_clicked(self):

        # (1) update drawing mode
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            self.drawing_mode = DrawingMode.GTVT_PEN

        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_PEN

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVT_PEN

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVN_PEN

        # (2) update widgets
        if (
            self.obs_study_gtvt_step
            in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            self._text_label["eraser.size"].hide()
            self._slider["eraser.size"].hide()
            self._text_label["pen.size"].show()
            self._slider["pen.size"].show()

    def _init_widgets_todo_list(self):
        todo_list_names = [
            TodoListLabel.SELECT_PATIENT,
            TodoListLabel.CLICK_GTVT_CENTER,
            TodoListLabel.DELINEATE_GTVT,
            TodoListLabel.DELINEATE_GTVT_TRANSVERSE,
            TodoListLabel.DELINEATE_GTVT_CORONAL,
            TodoListLabel.DELINEATE_GTVT_SAGITTAL,
            TodoListLabel.CLICK_GTVN_CENTERS,
            TodoListLabel.WAIT_GTVT_PRED,
            TodoListLabel.WAIT_GTVN_PRED,
            TodoListLabel.CORRECT_GTVT,
            TodoListLabel.CORRECT_GTVN,
        ]

        # init TodoListLabel
        for i in todo_list_names:
            # create todo list label
            self._text_label[i] = TodoListLabel(name=i)
            # set init state
            if i == TodoListLabel.SELECT_PATIENT:
                self._text_label[i].set_status_active()
            else:
                self._text_label[i].set_status_not_start()

        # v layout
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.setSpacing(2 if g.is_linux() else 2)

        next_btn_pixmap = QtGui.QPixmap(
            os.path.join(g.PROJ_DIR, "icons", "next_step.png")
        )
        for i in self._text_label.keys():
            if i in todo_list_names:
                # Create a grid layout to overlap the button and label
                grid = QGridLayout()

                # set todo list labels height
                self._text_label[i].setFixedHeight(27 if g.is_linux() else 40)

                # Add the QLabel to the grid layout spanning all columns
                # -1 makes it span all columns
                grid.addWidget(self._text_label[i], 0, 0, 1, -1)

                # next step buttons, right-aligned and overlapping the label
                if i in [
                    TodoListLabel.CLICK_GTVT_CENTER,
                    TodoListLabel.DELINEATE_GTVT,
                    TodoListLabel.CLICK_GTVN_CENTERS,
                    TodoListLabel.CORRECT_GTVT,
                    TodoListLabel.CORRECT_GTVN,
                ]:
                    self._btn[i] = QtWidgets.QPushButton()
                    # Set button height to match QLabel
                    # height-=4, because TodoListLabel has a 2px border
                    btn_w = btn_h = self._text_label[i].sizeHint().height() - 4
                    # margin-right: -2px, because TodoListLabel has a 2px border
                    self._btn[i].setStyleSheet(
                        "QPushButton { border: none; margin-right: 2px; padding: 0px; }"
                    )
                    # width is 2px larger than height, because position is adjusts 2px left
                    self._btn[i].setFixedSize(QSize(btn_w + 2, btn_h))
                    next_btn_pixmap = next_btn_pixmap.scaled(
                        btn_w, btn_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                    )
                    icon = QtGui.QIcon(next_btn_pixmap)
                    self._btn[i].setIconSize(QSize(btn_w, btn_h))
                    self._btn[i].setIcon(icon)

                    # hide button after init
                    self._btn[i].hide()

                    # Add the button to the same cell as the QLabel, with right alignment
                    grid.addWidget(self._btn[i], 0, 0, 1, -1, alignment=Qt.AlignRight)

                # Set stretch factors to adjust dynamically
                grid.setColumnStretch(0, 1)  # Ensures the QLabel expands properly

                # Add the grid layout to the vertical layout
                v_layout.addLayout(grid)

        # connect next btns to functions
        self._btn[TodoListLabel.CLICK_GTVT_CENTER].clicked.connect(
            self.__confirm_gtvt_center
        )
        self._btn[TodoListLabel.DELINEATE_GTVT].clicked.connect(
            self.__confirm_gtvt_delineation
        )
        self._btn[TodoListLabel.CLICK_GTVN_CENTERS].clicked.connect(
            self.__confirm_gtvn_centers
        )
        self._btn[TodoListLabel.CORRECT_GTVT].clicked.connect(
            self.__approve_gtvt_correction
        )
        self._btn[TodoListLabel.CORRECT_GTVN].clicked.connect(
            self.__approve_gtvn_correction
        )

        # container
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)

        # create qcollapsible space
        self._collap["todo.list"] = QCollapsible("TODO LIST")
        self._collap["todo.list"].addWidget(container)
        self._collap["todo.list"].expand()

    def __cleanup_future_step_3d_imgs(self):
        imgs_to_delete = []

        # cleanup all gtvn arrays
        if self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            # clear gtvn clicks np array, only keep crosses on the img frame
            imgs_to_delete += [
                "gtvn.clicks",
                "gtvn.pred",
                "gtvn.correction",
                "gtvn.correction.mask",
                "gtvn.pred.final",
            ]

        # clean up gtvt pred and correction arrays
        if self.obs_study_gtvt_step in [
            ObsStudyGTVtStep.DELINEATE,
            ObsStudyGTVtStep.CLICK_CENTER,
        ]:
            # dont clear the gtvt delineations np array
            imgs_to_delete += [
                "gtvt.pred",
                "gtvt.correction",
                "gtvt.correction.mask",
                "gtvt.pred.final",
            ]

        # cleanup gtvt click and delineations arrays
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
            # clear gtvt click np array, only keep cross on the img frame
            imgs_to_delete += [
                "gtvt.click",
                "gtvt.delineation.{}".format(Plane.TRANSVERSE),
                "gtvt.delineation.{}".format(Plane.CORONAL),
                "gtvt.delineation.{}".format(Plane.SAGITTAL),
            ]

        # cleanup arrays
        for i in imgs_to_delete:
            self.img_3d[i] = None

        # Initialize an empty array if GTVt delineation is None
        # This only occurs during the transition from
        # ObsStudyGTVtStep.CLICK_CENTER to ObsStudyGTVtStep.DELINEATE
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                if self.img_3d["gtvt.delineation.{}".format(plane)] is None:
                    self.img_3d["gtvt.delineation.{}".format(plane)] = np.zeros_like(
                        self.img_3d[Modal.CT]
                    )

    def __cleanup_future_step_files(self, keep_gtvn_clicks_nii: bool = True):
        # cleanup gtvn output files
        if self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            # idl gtvn training dir
            idl_gtvn_dir = os.path.join(
                g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id["gtvn"]
            )
            # delete gtvn nii files
            cross_valid_dir = os.path.join(
                idl_gtvn_dir,
                "patients",
                "patient={}".format(self._cur_patient),
                "round=01",
            )
            gtvn_files_list = [
                "gtvn_pred.nii.gz",
                "gtvn_distance_map.nii.gz",
                "gtvn_correction.nii.gz",
                "gtvn_correction_mask.nii.gz",
            ]
            if not keep_gtvn_clicks_nii:
                gtvn_files_list.append("gtvn_clicks.nii.gz")
            # keep "gtvn_clicks.nii.gz"
            for file_name in gtvn_files_list:
                g.delete_path(os.path.join(cross_valid_dir, file_name))

            # delete nii of each fold
            fold_dir_list = g.get_sub_dirs(
                idl_gtvn_dir, key_word="fold=", full_path=True
            )
            for fold_dir in fold_dir_list:
                epoch_dir = g.get_sub_dirs(fold_dir, key_word="epoch=", full_path=True)
                patient_dir = os.path.join(
                    epoch_dir[0], "patients", "patient={}".format(self._cur_patient)
                )
                g.delete_path(patient_dir)

        # cleanup gtvt output files
        if self.obs_study_gtvt_step in [
            ObsStudyGTVtStep.DELINEATE,
            ObsStudyGTVtStep.CLICK_CENTER,
        ]:
            idl_gtvt_dir = os.path.join(
                g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id["gtvt"]
            )
            g.delete_path(os.path.join(idl_gtvt_dir, "loss.png"))
            cur_patient_dir = os.path.join(
                idl_gtvt_dir, "patients", "patient={}".format(self._cur_patient)
            )
            g.delete_path(os.path.join(cur_patient_dir, "loss.json"))
            cur_round_dir = os.path.join(cur_patient_dir, "round=01")
            # keep gtvt pred/correction and cnn
            for file_name in [
                "gtvt_pred.nii.gz",
                "gtvt_correction.nii.gz",
                "gtvt_correction_mask.nii.gz",
                "round=01.pt",
            ]:
                g.delete_path(os.path.join(cur_round_dir, file_name))

        # cleanup gtvt delineations nii files
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
            cur_round_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id["gtvt"],
                "patients",
                "patient={}".format(self._cur_patient),
                "round=01",
            )
            # (1) keep "gtvt_click.nii.gz"
            # (2) NEVER delete "selected_slices.json",
            # otherwise SelectScenario will be GRAVITY_CENTER
            # and new selected slices will be generated
            # reclick gtvt center will regenerate "selected_slices.json",
            for file_name in [
                "gtvt_delineation_{}.nii.gz".format(Plane.TRANSVERSE),
                "gtvt_delineation_{}.nii.gz".format(Plane.CORONAL),
                "gtvt_delineation_{}.nii.gz".format(Plane.SAGITTAL),
                "gtvt_delineation.nii.gz",
            ]:
                g.delete_path(os.path.join(cur_round_dir, file_name))

    def __goto_click_gtvt_center(self):
        # (1) stop gtvt process and thread
        self.__stop_obs_study_gtvt_process()

        # (2) update status
        self.__update_obs_study_step(obs_study_gtvt_step=ObsStudyGTVtStep.CLICK_CENTER)

        # (3) update gtvt click pos
        # DO NOT clear self.gtvn_clicks_pos_3d
        if self.img_3d["gtvt.click"] is not None:
            # save the gtvt click pos to refresh the cross
            d, h, w = np.where(self.img_3d["gtvt.click"] == 1)
            d, h, w = int(d), int(h), int(w)
            self.gtvt_click_pos_3d = d, h, w

        # (4) clear future step 3d imgs and related files
        self.__cleanup_future_step_3d_imgs()
        self.__cleanup_future_step_files()

        # (5) refresh todolist, imgs and crosses
        self.__refresh_todo_list()
        self.refresh_imgs()
        self.refresh_crosses()

        # (6) update widgets
        self.refresh_mouse_cursor()
        self.__disable_annotation_tools()
        self.__refresh_next_btns()

        # (7) end and start timer
        self.__timer[ObsStudyTimer.DELINEATE_GTVT].end()
        self.__timer[ObsStudyTimer.WAIT_GTVT_PRED].end()
        self.__timer[ObsStudyTimer.CORRECT_GTVT].end()
        self.__timer[ObsStudyTimer.CLICK_GTVT_CENTER].start()

    def __refresh_next_btns(self):
        for i in [
            TodoListLabel.CLICK_GTVT_CENTER,
            TodoListLabel.DELINEATE_GTVT,
            TodoListLabel.CLICK_GTVN_CENTERS,
            TodoListLabel.CORRECT_GTVT,
            TodoListLabel.CORRECT_GTVN,
        ]:
            self._btn[i].hide()

        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
            self._btn[TodoListLabel.CLICK_GTVT_CENTER].show()

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            self._btn[TodoListLabel.DELINEATE_GTVT].show()

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            self._btn[TodoListLabel.CLICK_GTVN_CENTERS].show()

        else:
            if self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
                self._btn[TodoListLabel.CORRECT_GTVT].show()
            if self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
                self._btn[TodoListLabel.CORRECT_GTVN].show()

    # this function is connected to widget, dont set input params to this function
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
        g.create_dir(cur_round_dir)
        idl_gtvt_click = self.img_3d["gtvt.click"].copy()
        # flip left/right
        idl_gtvt_click = np.flip(idl_gtvt_click, axis=2)
        # turn upside down
        idl_gtvt_click = np.flip(idl_gtvt_click, axis=0)
        # save nii
        g.save_nii(
            img=idl_gtvt_click,
            save_path=os.path.join(cur_round_dir, "gtvt_click.nii.gz"),
            spacing=g.NII_SPACING,
        )

        # (4) save gtvt selected_slices.json
        pos = np.where(idl_gtvt_click == 1)
        selected_slices = Dict()
        selected_slices[Plane.TRANSVERSE]["round=01"] = List(pos[0]).to_str()
        selected_slices[Plane.CORONAL]["round=01"] = List(pos[1]).to_str()
        selected_slices[Plane.SAGITTAL]["round=01"] = List(pos[2]).to_str()
        # NEVER delete "selected_slices.json",
        # otherwise SelectScenario will be GRAVITY_CENTER and
        # new selected slices will be generated
        # reclick gtvt center will regenerate "selected_slices.json",
        g.save_json(
            data=selected_slices,
            path=os.path.join(cur_patient_dir, "selected_slices.json"),
        )

        # (5) end timer
        self.__timer[ObsStudyTimer.CLICK_GTVT_CENTER].end()

        # (6) goto next step
        self.__goto_delineate_gtvt()

    def __goto_delineate_gtvt(self):
        # (1) stop gtvt process and thread
        self.__stop_obs_study_gtvt_process()

        # (2) update status
        self.__update_obs_study_step(obs_study_gtvt_step=ObsStudyGTVtStep.DELINEATE)
        self.drawing_mode = DrawingMode.GTVT_PEN

        # (3) clear future step 3d imgs and related files
        # DO NOT clear self.gtvt_click_pos_3d
        # DO NOT clear self.gtvn_clicks_pos_3d
        self.__cleanup_future_step_3d_imgs()
        self.__cleanup_future_step_files()

        # (4) refresh todolist and imgs, delete crosses
        self.__refresh_todo_list()
        self.refresh_imgs()  # after __refresh_todo_list()
        self.delete_all_crosses()

        # (5) update widgets
        self.__enable_annotation_tools()
        self.__refresh_next_btns()

        # (6) start recording time
        self.__timer[ObsStudyTimer.CLICK_GTVT_CENTER].end()
        self.__timer[ObsStudyTimer.WAIT_GTVT_PRED].end()
        self.__timer[ObsStudyTimer.CORRECT_GTVT].end()
        self.__timer[ObsStudyTimer.DELINEATE_GTVT].start()

    def __stop_obs_study_gtvt_process(self):
        # stop progress thread
        self.__obs_study_gtvt_progress_thread.stop()

        # terminate gtvt training process
        if self._obs_study_gtvt_process is not None:
            if self._obs_study_gtvt_process.is_alive():
                self._obs_study_gtvt_process.terminate()
            # Clean up process resources
            self._obs_study_gtvt_process.join()

    def __start_obs_study_gtvt_process(
        self, idl_gtvt_id, dataset_ver, patient, queue, debug_mode
    ):
        # Start the training process in a separate process
        process_data = idl_gtvt_id, dataset_ver, patient, queue, debug_mode

        self._obs_study_gtvt_process = Process(
            target=self._obs_study_gtvt_process_func, args=process_data
        )
        self._obs_study_gtvt_process.start()

        # Start the thread to monitor progress
        self.__obs_study_gtvt_progress_thread.queue = queue  # Set the queue
        if not self.__obs_study_gtvt_progress_thread.isRunning():
            self.__obs_study_gtvt_progress_thread.start()

    @staticmethod
    def _obs_study_gtvt_process_func(
        idl_gtvt_id, dataset_ver, patient, queue, debug_mode
    ):
        try:
            training = IDLGTVtTraining()
            training.obs_study(
                idl_gtvt_id=idl_gtvt_id,
                dataset_ver=dataset_ver,
                patient=patient,
                queue=queue,
                device_id=1,  # Use card 1, as GTVt re-training requires more resources than GTVn inference.
                debug_mode=debug_mode,
            )
        except Exception:
            pass

    # this function is connected to widget, dont set input params to this function
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
        g.create_dir(cur_round_dir)
        # create an empty merge delineation array
        gtvt_delineation_merged = np.zeros_like(self.img_3d[Modal.CT])

        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            gtvt_delineation_cur_plane = self.img_3d[
                "gtvt.delineation.{}".format(plane)
            ].copy()
            # flip left/right
            gtvt_delineation_cur_plane = np.flip(gtvt_delineation_cur_plane, axis=2)
            # turn upside down
            gtvt_delineation_cur_plane = np.flip(gtvt_delineation_cur_plane, axis=0)
            # save nii
            g.save_nii(
                img=gtvt_delineation_cur_plane,
                save_path=os.path.join(
                    cur_round_dir, "gtvt_delineation_{}.nii.gz".format(plane)
                ),
                spacing=g.NII_SPACING,
            )
            # merge
            gtvt_delineation_merged = np.maximum(
                gtvt_delineation_merged,
                gtvt_delineation_cur_plane,
            )
        # save merged gtvt delineation for iDL GTVt training
        g.save_nii(
            img=gtvt_delineation_merged,
            save_path=os.path.join(cur_round_dir, "gtvt_delineation.nii.gz"),
            spacing=g.NII_SPACING,
        )

        # (3) start gtvt process
        self.__start_obs_study_gtvt_process(
            self._idl_id["gtvt"],
            self.dataset_ver,
            self._cur_patient,
            self.__queue,
            self._debug_mode,
        )
        # self.__obs_study_gtvt_thread.set_param(
        #     idl_gtvt_id=self._idl_id["gtvt"],
        #     dataset_ver=self.dataset_ver,
        #     patient=self._cur_patient,
        #     debug_mode=self._debug_mode,
        # )
        # self.__obs_study_gtvt_thread.start()

        # (4) end and start timer
        self.__timer[ObsStudyTimer.DELINEATE_GTVT].end()
        self.__timer[ObsStudyTimer.WAIT_GTVT_PRED].start()

        # (5) goto next step
        if self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            self.__goto_click_gtvn_centers()
        else:
            self.__goto_correct_pred(obs_study_gtvt_step=ObsStudyGTVtStep.WAIT_PRED)

    def __goto_click_gtvn_centers(self):
        # (1) stop idl gtvn qthread
        self.__obs_study_gtvn_thread.stop()

        # (2) update status (before refresh images)
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            obs_study_gtvt_step = ObsStudyGTVtStep.WAIT_PRED
        else:
            obs_study_gtvt_step = None
        self.__update_obs_study_step(
            obs_study_gtvt_step=obs_study_gtvt_step,
            obs_study_gtvn_step=ObsStudyGTVnStep.CLICK_CENTERS,
        )

        # (3) update gtvn clicks
        if self.img_3d["gtvn.clicks"] is not None:
            # save pos of gtvn clicks to refresh the cross
            pos = np.where(self.img_3d["gtvn.clicks"] == 1)
            self.gtvn_clicks_pos_3d = List(zip(*pos))

        # (4) clear future step 3d imgs and related files, then combine images
        self.__cleanup_future_step_3d_imgs()
        self.__cleanup_future_step_files()
        self.__combine_pred_delineation_correction()

        # (5) refresh todolist, imgs and crosses
        self.__refresh_todo_list()
        self.refresh_imgs()
        self.refresh_crosses()

        # (6) update widgets
        self.refresh_mouse_cursor()
        self.__disable_annotation_tools()
        self.__refresh_next_btns()

        # (7) end and start timer
        self.__timer[ObsStudyTimer.WAIT_GTVN_PRED].end()
        self.__timer[ObsStudyTimer.CORRECT_GTVN].end()
        self.__timer[ObsStudyTimer.CLICK_GTVN_CENTERS].start()

    # this function is connected to widget, dont set input params to this function
    def __approve_gtvt_correction(self):
        # self.__approve_correction("gtvt")
        self.__goto_correct_pred(obs_study_gtvt_step=ObsStudyGTVtStep.APPROVED)

    # this function is connected to widget, dont set input params to this function
    def __approve_gtvn_correction(self):
        # self.__approve_correction("gtvn")
        self.__goto_correct_pred(obs_study_gtvn_step=ObsStudyGTVnStep.APPROVED)

    def __goto_correct_pred(
        self,
        obs_study_gtvt_step: str = None,
        obs_study_gtvn_step: str = None,
    ):
        # (1) check obs_study_gtvt_step
        if obs_study_gtvt_step not in [
            ObsStudyGTVtStep.WAIT_PRED,
            ObsStudyGTVtStep.CORRECT,
            ObsStudyGTVtStep.APPROVED,
            None,
        ]:
            g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        # (2) check obs_study_gtvn_step
        if obs_study_gtvn_step not in [
            ObsStudyGTVnStep.WAIT_PRED,
            ObsStudyGTVnStep.CORRECT,
            ObsStudyGTVnStep.APPROVED,
            None,
        ]:
            g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        # (3) update obs study steps
        if obs_study_gtvt_step is not None:
            self.__update_obs_study_step(obs_study_gtvt_step=obs_study_gtvt_step)
        if obs_study_gtvn_step is not None:
            self.__update_obs_study_step(obs_study_gtvn_step=obs_study_gtvn_step)

        # (4) switch drawing mode (before update widgets)
        # only gtvt is being corrected
        if (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.__switch_gtv_drawing_mode()
        # only gtvn is being corrected
        elif (
            self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
            and self.obs_study_gtvt_step != ObsStudyGTVtStep.CORRECT
        ):
            self._radio_btn["correct.gtvn"].setChecked(True)
            self.__switch_gtv_drawing_mode()
        # both gtvt and gtvn are being corrected
        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            # only triggered when TodoListLabel.CORRECT_GTVT is clicked
            if obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
                self._radio_btn["correct.gtvt"].setChecked(True)
                self.__switch_gtv_drawing_mode()
            # only triggered when TodoListLabel.CORRECT_GTVN is clicked
            elif obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
                self._radio_btn["correct.gtvn"].setChecked(True)
                self.__switch_gtv_drawing_mode()
            else:
                pass
        else:
            pass

        # (5) save corrections and correction masks
        # gtvt
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED:
            self.__save_corrections_and_masks("gtvt")
        else:
            pass
        # gtvn
        if self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED:
            self.__save_corrections_and_masks("gtvn")
        else:
            pass

        # (6) update 3d arrays for display
        # init correction and mask if they are None
        for gtv in ["gtvt", "gtvn"]:
            for i in ["{}.correction".format(gtv), "{}.correction.mask".format(gtv)]:
                if self.img_3d[i] is None:
                    self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])
        # combine 3d arrays
        self.__combine_pred_delineation_correction()

        # (7) refresh todolist and imgs
        self.__refresh_todo_list()
        self.refresh_imgs()

        # (8) update widgets
        self.__refresh_next_btns()
        self.refresh_mouse_cursor()
        if (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            self.__enable_annotation_tools()
        else:
            self.__disable_annotation_tools()

        # (9) end / start gtvt timer
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            self.__timer[ObsStudyTimer.CORRECT_GTVT].start()
        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED:
            self.__timer[ObsStudyTimer.CORRECT_GTVT].end()
        else:
            pass

        # (10) end / start gtvn timer
        if self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
            self.__timer[ObsStudyTimer.CORRECT_GTVN].start()
        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED:
            self.__timer[ObsStudyTimer.CORRECT_GTVN].end()
        else:
            pass

        # (11) end total timer
        if (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED
        ):
            self.__timer[ObsStudyTimer.PATIENT_TOTAL_TIME].end()
        else:
            pass

    def on_todo_list_clicked(self, todo_list_label: TodoListLabel):
        # (1) jump to step SELECT_PATIENT
        if todo_list_label == self._text_label[TodoListLabel.SELECT_PATIENT]:
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
        elif todo_list_label == self._text_label[TodoListLabel.CLICK_GTVT_CENTER]:
            # can not jump to DELINEATE_GTVT from the following step:
            if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
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

            self.__goto_click_gtvt_center()

        # (3) jump to step DELINEATE_GTVT
        elif todo_list_label in [
            self._text_label[TodoListLabel.DELINEATE_GTVT],
            self._text_label[TodoListLabel.DELINEATE_GTVT_TRANSVERSE],
            self._text_label[TodoListLabel.DELINEATE_GTVT_CORONAL],
            self._text_label[TodoListLabel.DELINEATE_GTVT_SAGITTAL],
        ]:
            # can not jump to DELINEATE_GTVT from the following steps:
            if self.obs_study_gtvt_step in [
                ObsStudyGTVtStep.CLICK_CENTER,
                ObsStudyGTVtStep.DELINEATE,
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

            self.__goto_delineate_gtvt()

        # (4) jump to step CLICK_GTVN_CENTER
        elif todo_list_label == self._text_label[TodoListLabel.CLICK_GTVN_CENTERS]:
            # can not jump to CLICK_GTVN_CENTER from the following steps:
            if (
                self.obs_study_gtvt_step
                in [
                    ObsStudyGTVtStep.CLICK_CENTER,
                    ObsStudyGTVtStep.DELINEATE,
                ]
                or self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS
            ):
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

            self.__goto_click_gtvn_centers()

        # (5) revert to step CORRECT_GTVT
        elif todo_list_label == self._text_label[TodoListLabel.CORRECT_GTVT]:
            if self.obs_study_gtvt_step != ObsStudyGTVtStep.APPROVED:
                return

            text = "Would you like to revert to SETP 6 and re-correct GTVt predictions?"
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

            # goto correct prediction
            self.__goto_correct_pred(obs_study_gtvt_step=ObsStudyGTVtStep.CORRECT)

        # (6) revert to step CORRECT_GTVT/GTVN
        elif todo_list_label == self._text_label[TodoListLabel.CORRECT_GTVN]:
            if self.obs_study_gtvn_step != ObsStudyGTVnStep.APPROVED:
                return

            text = "Would you like to revert to SETP 6 and re-correct GTVn predictions?"
            reply = QMessageBox.question(
                self,
                "Message",
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

            # goto correct prediction
            self.__goto_correct_pred(obs_study_gtvn_step=ObsStudyGTVnStep.CORRECT)

        else:
            pass

    # this function is connected to widget, dont set input params to this function
    def __on_btn_eraser_clicked(self):

        # (1) update drawing mode
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            self.drawing_mode = DrawingMode.GTVT_ERASER

        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_ERASER
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_ERASER

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVT_ERASER

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVN_ERASER

        # (2) update widgets
        if (
            self.obs_study_gtvt_step
            in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            self._text_label["pen.size"].hide()
            self._slider["pen.size"].hide()
            self._text_label["eraser.size"].show()
            self._slider["eraser.size"].show()

    def __update_obs_study_gtvt_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        g.clamp_value(progress_int, (0, 100))
        self.__progress_bar["gtvt"].setValue(progress_int)

    def __on_obs_study_gtvt_thread_finished(self):
        # (1) update obs_study_gtvt_step
        self.__update_obs_study_step(obs_study_gtvt_step=ObsStudyGTVtStep.CORRECT)

        # (2) load and combine 3d imgs
        self._load_idl_gtvt_data()
        self.__combine_pred_delineation_correction()
        # init correction and mask
        # correction and mask are empty anyway,
        # its efficient to init them after __combine_pred_delineation_correction
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
        self.__refresh_next_btns()

        # user is not clicking or correcting gtvn
        if self.obs_study_gtvn_step in [
            ObsStudyGTVnStep.WAIT_PRED,
            ObsStudyGTVnStep.APPROVED,
        ]:
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.drawing_mode = DrawingMode.GTVT_PEN
            # change mouse cursor after:
            # (1) obs study steps updated
            # (2) drawing mode updated
            self.refresh_mouse_cursor()

        # otherwise, do nothing to avoid interrupting user input
        else:
            pass

        # (5) end and start timer
        self.__timer[ObsStudyTimer.WAIT_GTVT_PRED].end()
        if self.drawing_mode in [
            DrawingMode.GTVT_PEN,
            DrawingMode.GTVT_ERASER,
            DrawingMode.GTVT_CLEAR,
            DrawingMode.GTVT_RESTORE,
        ]:
            self.__timer[ObsStudyTimer.CORRECT_GTVT].start()

    # this function is connected to widget, dont set input params to this function
    def __confirm_gtvn_centers(self):
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
            # user decided to re-click gtvn center
            if reply == QMessageBox.No:
                return
            # user confirm there is no gtvn
            else:
                # Clean up GTVn arrays and files of future steps
                # Note: obs_study_gtvn_step is still CLICK_CENTERS,
                # ensuring that the arrays and files are cleaned up correctly.
                self.__cleanup_future_step_3d_imgs()
                # also delete "gtvn_clicks.nii.gz"
                self.__cleanup_future_step_files(keep_gtvn_clicks_nii=False)
                self.__timer[ObsStudyTimer.CLICK_GTVN_CENTERS].end()
                self.__timer[ObsStudyTimer.WAIT_GTVN_PRED].start()
                # gtvn step will be updated in the following function
                self.__on_obs_study_gtvn_thread_finished()
                return

        # (2) update status
        self.__update_obs_study_step(obs_study_gtvn_step=ObsStudyGTVnStep.WAIT_PRED)

        # (3) add clicks into 3d img
        # if self.img_3d["gtvn.clicks"] is None:
        self.img_3d["gtvn.clicks"] = np.zeros_like(self.img_3d[Modal.CT])
        for pos in self.gtvn_clicks_pos_3d:
            # pos 0-transverse 1-coronal 2-saggital
            self.img_3d["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1

        # (4) transform gtvn clicks ndarray for idl.gtvn thread, also save the nii
        cur_round_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id["gtvn"],
            "patients",
            "patient={}".format(self._cur_patient),
            "round=01",
        )
        g.create_dir(cur_round_dir)
        # copy data (dont change origin ndarray)
        idl_gtvn_clicks = self.img_3d["gtvn.clicks"].copy()
        # only flip non-empty img
        if idl_gtvn_clicks.max() > 0:
            # flip left/right
            idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=2)
            # turn upside down
            idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=0)
            # save nii
            g.save_nii(
                img=idl_gtvn_clicks,
                save_path=os.path.join(cur_round_dir, "gtvn_clicks.nii.gz"),
                spacing=g.NII_SPACING,
            )

        # (5) start idl gtvn thread
        self.__obs_study_gtvn_thread.set_param(
            idl_gtvn_id=self._idl_id["gtvn"],
            dataset_ver=self.dataset_ver,
            patient=self._cur_patient,
            idl_gtvn_clicks=idl_gtvn_clicks,
            debug_mode=self._debug_mode,
        )
        self.__obs_study_gtvn_thread.start()

        # (6) refresh todolist and imgs, delete crosses
        self.__refresh_todo_list()
        # refresh contours only (on all img frames)
        self.refresh_imgs(
            reload_origin_rgb=False,
            reload_zoomed_rgb=False,
        )
        self.delete_all_crosses()

        # (7) update widgets
        self.__refresh_next_btns()
        # gtvt training is waiting for correction,
        # enable annotation tools and switch to gtvt pen mode
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.drawing_mode = DrawingMode.GTVT_PEN
            self.__enable_annotation_tools()
        # gtvt training is still ongoing or gtvt correction is approved
        # disable annotation tools as there is nothing to be corrected
        elif self.obs_study_gtvt_step in [
            ObsStudyGTVtStep.WAIT_PRED,
            ObsStudyGTVtStep.APPROVED,
        ]:
            self.__disable_annotation_tools()

        # (8) start and end timer
        self.__timer[ObsStudyTimer.CLICK_GTVN_CENTERS].end()
        self.__timer[ObsStudyTimer.WAIT_GTVN_PRED].start()

    def __update_obs_study_gtvn_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        g.clamp_value(progress_int, (0, 100))
        self.__progress_bar["gtvn"].setValue(progress_int)

    def __on_obs_study_gtvn_thread_finished(self):
        # (1) update obs study step
        self.__update_obs_study_step(obs_study_gtvn_step=ObsStudyGTVnStep.CORRECT)

        # (2) load and combine 3d imgs
        self._load_idl_gtvn_data()
        self.__combine_pred_delineation_correction()
        # init correction and mask
        # correction and mask are empty anyway,
        # its efficient to init them after __combine_pred_delineation_correction
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
        self.__refresh_next_btns()

        # user is not clicking, delineating or correcting gtvt
        if self.obs_study_gtvt_step in [
            ObsStudyGTVtStep.WAIT_PRED,
            ObsStudyGTVtStep.APPROVED,
        ]:
            self._radio_btn["correct.gtvn"].setChecked(True)
            self.drawing_mode = DrawingMode.GTVN_PEN
            # change mouse cursor after:
            # (1) obs study steps updated
            # (2) drawing mode updated
            self.refresh_mouse_cursor()

        # otherwise, do nothing to avoid interrupting user input
        else:
            pass

        # (5) end and start timer
        self.__timer[ObsStudyTimer.WAIT_GTVN_PRED].end()
        if self.drawing_mode in [
            DrawingMode.GTVN_PEN,
            DrawingMode.GTVN_ERASER,
            DrawingMode.GTVN_CLEAR,
            DrawingMode.GTVN_RESTORE,
        ]:
            self.__timer[ObsStudyTimer.CORRECT_GTVN].start()

    # check delineation in 3 different planes
    def __update_gtvt_delineated_status(self) -> Dict:
        # no gtvt click
        if self.img_3d["gtvt.click"] is None:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                self.__gtvt_delineated_state[plane] = False
                self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()
            return

        d, h, w = np.where(self.img_3d["gtvt.click"] == 1)
        d, h, w = int(d), int(h), int(w)

        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            gtvt_delineation_3d = self.img_3d["gtvt.delineation.{}".format(plane)]

            if gtvt_delineation_3d is None:
                self.__gtvt_delineated_state[plane] = False
                self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()

            else:
                if plane == Plane.TRANSVERSE:
                    gtvt_delineation_2d = gtvt_delineation_3d[d, :, :]
                elif plane == Plane.CORONAL:
                    gtvt_delineation_2d = gtvt_delineation_3d[:, h, :]
                elif plane == Plane.SAGITTAL:
                    gtvt_delineation_2d = gtvt_delineation_3d[:, :, w]

                if gtvt_delineation_2d.max() <= 0:
                    self.__gtvt_delineated_state[plane] = False
                    self._text_label[
                        "delineate.gtvt.{}".format(plane)
                    ].set_status_missing()
                else:
                    self.__gtvt_delineated_state[plane] = True
                    self._text_label[
                        "delineate.gtvt.{}".format(plane)
                    ].set_status_completed()

    def __change_color(self, pixmap: QtGui.QPixmap, old_color, new_color):
        image = pixmap.toImage()
        old_qcolor = QtGui.QColor(*old_color)  # Unpack the tuple
        new_qcolor = QtGui.QColor(*new_color)  # Unpack the tuple

        for x in range(image.width()):
            for y in range(image.height()):
                if image.pixelColor(x, y) == old_qcolor:
                    image.setPixelColor(x, y, new_qcolor)
        return QtGui.QPixmap.fromImage(image)

    def refresh_mouse_cursor(self):
        if (
            self.obs_study_gtvt_step
            not in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            self.setCursor(Qt.ArrowCursor)
            return

        if self._under_mouse_img_frame_name() is None:
            self.setCursor(Qt.ArrowCursor)
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

        # get cursor name
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            cursor_name = TodoListLabel.DELINEATE_GTVT

        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self.drawing_mode in [
                DrawingMode.GTVT_PEN,
                DrawingMode.GTVT_ERASER,
                DrawingMode.GTVT_CLEAR,
                DrawingMode.GTVT_RESTORE,
            ]:
                cursor_name = TodoListLabel.CORRECT_GTVT
            elif self.drawing_mode in [
                DrawingMode.GTVN_PEN,
                DrawingMode.GTVN_ERASER,
                DrawingMode.GTVN_CLEAR,
                DrawingMode.GTVN_RESTORE,
            ]:
                cursor_name = TodoListLabel.CORRECT_GTVN

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            cursor_name = TodoListLabel.CORRECT_GTVT

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
            cursor_name = TodoListLabel.CORRECT_GTVN

        else:
            g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        cursor_pixmap = self.__cursor[cursor_name][tool]
        self.setCursor(QtGui.QCursor(cursor_pixmap, left, top))

    def _init_widgets_cursor(self):
        self.__cursor = Dict()
        cursor_size = 32  # cursor size is no larger than 32
        origin_color = (0, 0, 0)
        for tool in ["pen", "eraser", "clear", "restore"]:
            origin_cursor = QtGui.QPixmap(
                (os.path.join(g.PROJ_DIR, "icons", "{}_cursor.png".format(tool)))
            )
            for name in [
                TodoListLabel.DELINEATE_GTVT,
                TodoListLabel.CORRECT_GTVT,
                TodoListLabel.CORRECT_GTVN,
            ]:
                self.__cursor[name][tool] = origin_cursor.scaled(
                    cursor_size,
                    cursor_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                # change color (after cursor pixmap is scaled to 32*32)
                # as __change_color is not efficiency
                if name == TodoListLabel.DELINEATE_GTVT:
                    new_color = self.color["gtvt.delineation"]
                elif name == TodoListLabel.CORRECT_GTVT:
                    new_color = self.color["gtvt.pred"]
                elif name == TodoListLabel.CORRECT_GTVN:
                    new_color = self.color["gtvn.pred"]
                self.__cursor[name][tool] = self.__change_color(
                    pixmap=self.__cursor[name][tool],
                    old_color=origin_color,
                    new_color=new_color,
                )

    def _init_color(self, ui_settings: Dict):
        super()._init_color(ui_settings)
        self.color["eraser"] = self.color["black"]  # transparent
        self.color["gtvt.correction"] = self.color["gtvt.pred"]
        self.color["gtvn.correction"] = self.color["gtvn.pred"]
        self.color["gtvt.pred.final"] = self.color["gtvt.pred"]
        self.color["gtvn.pred.final"] = self.color["gtvn.pred"]
        self.color["gtvt.click"] = self.color[
            ui_settings["contour.color"]["gtvt"]["click.obs.study"]
        ]
        self.color["gtvn.clicks"] = self.color[
            ui_settings["contour.color"]["gtvn"]["clicks.obs.study"]
        ]
        self.color["gtvt.delineation"] = self.color[
            ui_settings["contour.color"]["gtvt"]["delineation.obs.study"]
        ]

    # this function is connected to widget, dont set input params to this function
    def __on_btn_restore_clicked(self):
        if (
            self.obs_study_gtvt_step != ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            return

        # update drawing mode
        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_RESTORE
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_RESTORE

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVT_RESTORE

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVN_RESTORE

    # this function is connected to widget, dont set input params to this function
    def __on_btn_clear_clicked(self):
        if (
            self.obs_study_gtvt_step
            not in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            return

        # update drawing mode
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            self.drawing_mode = DrawingMode.GTVT_CLEAR

        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self._radio_btn["correct.gtvt"].isChecked():
                self.drawing_mode = DrawingMode.GTVT_CLEAR
            elif self._radio_btn["correct.gtvn"].isChecked():
                self.drawing_mode = DrawingMode.GTVN_CLEAR

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVT_CLEAR

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
            self.drawing_mode = DrawingMode.GTVN_CLEAR

    def __get_gtvt_center_slices_id(self):
        if self.gtvt_click_pos_3d is None:
            g.error_exit("self.gtvt_click_pos_3d is empty")
        else:
            center_slices_id = Dict()
            center_slices_id[Plane.TRANSVERSE] = self.gtvt_click_pos_3d[0]
            center_slices_id[Plane.CORONAL] = self.gtvt_click_pos_3d[1]
            center_slices_id[Plane.SAGITTAL] = self.gtvt_click_pos_3d[2]
            for i in center_slices_id.keys():
                center_slices_id[i] = int(center_slices_id[i])
        return center_slices_id

    def __get_gtvn_center_slices_id(self):
        if len(self.gtvn_clicks_pos_3d) == 0:
            g.error_exit("self.gtvn_clicks_pos_3d is empty")
        else:
            center_slices_id = Dict()
            center_slices_id[Plane.TRANSVERSE] = self.gtvn_clicks_pos_3d[-1][0]
            center_slices_id[Plane.CORONAL] = self.gtvn_clicks_pos_3d[-1][1]
            center_slices_id[Plane.SAGITTAL] = self.gtvn_clicks_pos_3d[-1][2]
            for i in center_slices_id.keys():
                center_slices_id[i] = int(center_slices_id[i])
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
        if (
            self.obs_study_gtvt_step != ObsStudyGTVtStep.CLICK_CENTER
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CLICK_CENTERS
        ):
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
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
            self.gtvt_click_pos_3d = None

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
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

    def __enable_annotation_tools(self):
        # annotation buttons
        for i in ["pen", "eraser", "clear"]:
            self._btn[i].setEnabled(True)

        if (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            self._btn["restore"].setEnabled(True)
        else:
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
            if (
                self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
                and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
            ):
                self._radio_btn[i].show()
            else:
                self._radio_btn[i].hide()

        if not self._collap["annotation"].isExpanded():
            self._collap["annotation"].expand()

    def __disable_annotation_tools(self):
        for i in ["pen", "eraser", "clear", "restore"]:
            self._btn[i].setEnabled(False)
        for i in ["pen.size", "eraser.size"]:
            self._text_label[i].hide()
            self._slider[i].hide()
        for i in ["correct.gtvt", "correct.gtvn"]:
            self._radio_btn[i].hide()

    def _init_widgets_annotation(self, ui_settings: Dict):

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
            # load setting
            min_size = int(ui_settings[i]["min"])
            min_size = max(min_size, 0)
            max_size = int(ui_settings[i]["max"])
            max_size = min(max_size, 100)
            min_size = min(min_size, max_size)

            # create slider
            self._slider[i] = QtWidgets.QSlider()
            self._slider[i].setFixedHeight(g.SLIDER_HEIGHT)
            self._slider[i].setOrientation(Qt.Horizontal)
            self._slider[i].hide()
            min_size *= 100
            self._slider[i].setMinimum(min_size)
            max_size *= 100
            self._slider[i].setMaximum(max_size)
            self._slider[i].setValue(min_size + (max_size - min_size) // 2)

        # drawing mode radio buttons
        self.__radio_group_drawing_mode = QtWidgets.QButtonGroup()
        for i in ["gtvt", "gtvn"]:
            self.__radio_group_drawing_mode.addButton(
                self._radio_btn["correct.{}".format(i)]
            )
            self._radio_btn["correct.{}".format(i)].hide()
        self.__radio_group_drawing_mode.buttonClicked.connect(
            self.__switch_gtv_drawing_mode
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

        # gtvt/gtvn thread (after progress bars and progress bar labels initialized)
        self._obs_study_gtvt_process = None
        self.__obs_study_gtvt_progress_thread = ObsStudyGTVtProgressThread(
            progress_bar=self.__progress_bar["gtvt"],
            progress_bar_label=self._text_label["gtvt.progress"],
        )
        self.__obs_study_gtvt_progress_thread.progress_signal.connect(
            self.__update_obs_study_gtvt_progress_bar
        )
        self.__obs_study_gtvt_progress_thread.complete_signal.connect(
            self.__on_obs_study_gtvt_thread_finished
        )
        self.__obs_study_gtvn_thread = ObsStudyGTVnThread(
            progress_bar=self.__progress_bar["gtvn"],
            progress_bar_label=self._text_label["gtvn.progress"],
        )
        self.__obs_study_gtvn_thread.progress_signal.connect(
            self.__update_obs_study_gtvn_progress_bar
        )
        self.__obs_study_gtvn_thread.complete_signal.connect(
            self.__on_obs_study_gtvn_thread_finished
        )

        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)
        self._collap["annotation"].addWidget(container)

    def _init_widgets(self, ui_settings: Dict):
        super()._init_widgets(ui_settings)

        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._collap[i].collapse()
            self._collap[i].hide()

        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].collapse()

    def _clear_img_3d(self):
        super()._clear_img_3d()
        for i in ["gtvt.correction.mask", "gtvn.correction.mask"]:
            self.img_3d[i] = None

    def _init_data(self, ui_settings: Dict):
        super()._init_data(ui_settings)

        # init baseline id and idl.gtvt/gtvn id, keep them unchanged
        self._baseline_id = "baseline_obs.study"

        # (1) new training
        # initlize idl.gtvt/gtvn id
        if self.__train_id == "Start a new experiment":
            cur_time = g.get_cur_time_str()
            for i in ["gtvt", "gtvn"]:
                self._idl_id[i] = "idl.{}_".format(i) + cur_time

                if self.__user_name != "" and self.__user_name is not None:
                    while self.__user_name.startswith("_"):
                        self.__user_name = self.__user_name[1:]
                    while self.__user_name.endswith("_"):
                        self.__user_name = self.__user_name[:-1]
                    self._idl_id[i] += "_" + self.__user_name

                if self._debug_mode:
                    self._idl_id[i] += "_" + g.DELETE_FLAG

            # create idl.gtvt/gtvn folders
            for i in ["gtvt", "gtvn"]:
                g.create_dir(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i]
                    )
                )

        # (2) existing train id
        else:
            for i in ["gtvt", "gtvn"]:
                self._idl_id[i] = "idl.{}_{}".format(i, self.__train_id)

        # initialize the position of gtvn clicks
        self.gtvn_clicks_pos_3d = List()
        # position of gtvt click is already initialized in super()._init_data() above

        # init drawing
        self.drawing_mode = DrawingMode.GTVT_PEN
        self.paint_pos = None  # Store the last painted point
        self.__gtvt_delineated_state = Dict()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.__gtvt_delineated_state[plane] = False

        # init obs study steps and json file
        self.obs_study_gtvt_step = None
        self.obs_study_gtvn_step = None
        self.__obs_study_step_json_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id["gtvt"],  # only save it in gtvt folder
            "obs_study_step.json",
        )
        if not os.path.exists(self.__obs_study_step_json_path):
            g.save_json({}, self.__obs_study_step_json_path)

        # load/save interpolation step
        interpolation_setting_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id["gtvt"],
            "interpolation.json",
        )
        if os.path.exists(interpolation_setting_path):
            self.interpolation_step = g.load_json(interpolation_setting_path)["step"]
            self.interpolation_step = max(1, int(self.interpolation_step))
        else:
            self.interpolation_step = ui_settings["interpolation.step"]
            self.interpolation_step = max(1, int(self.interpolation_step))
            g.save_json({"step": self.interpolation_step}, interpolation_setting_path)

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
    def __switch_gtv_drawing_mode(self):
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
            # switch timer
            self.__timer[ObsStudyTimer.CORRECT_GTVN].pause()
            self.__timer[ObsStudyTimer.CORRECT_GTVT].start()

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
            # switch timer
            self.__timer[ObsStudyTimer.CORRECT_GTVT].pause()
            self.__timer[ObsStudyTimer.CORRECT_GTVN].start()

    def get_pen_size(self):
        pen_size = self._slider["pen.size"].value() / 100
        pen_size *= self.get_zoomin_factor()
        return pen_size

    def get_eraser_size(self):
        eraser_size = self._slider["eraser.size"].value() / 100
        eraser_size *= self.get_zoomin_factor()
        return eraser_size

    def _load_baseline_data(self):
        # self._reset_zoomin()
        self._clear_img_3d()
        self._clear_img_frames()

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self.combox["patient"].setCurrentIndex(-1)  # show nothing

    def _display_instruction_on_top_left(self, qimg: QtGui.QImage):
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

        # click gtvt center
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
            text = "Please click the center of primary Gross Tumor Volumes (GTVt)"

        # delineate gtvt
        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            text = "Please delineate GTVt in 3 anatomical planes"

        # click gtvn center
        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            text = "Please click the center of malignant lymph nodes (GTVn)"

        # other conditions:
        # gtvt/gtvn - WAIT_PRED/CORRECT/APPROVED, (3*3=9 conditions)
        else:
            # (1) both waiting
            if (
                self.obs_study_gtvt_step == ObsStudyGTVtStep.WAIT_PRED
                and self.obs_study_gtvn_step == ObsStudyGTVnStep.WAIT_PRED
            ):
                text = "Neural Network is generating auto-segmentations, please wait..."

            # (2) both being corrected
            elif (
                self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
                and self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
            ):
                text = "Please correct the GTVt and GTVn auto-segmentations"

            # (3) both approved
            elif (
                self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED
                and self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED
            ):
                text = "Auto-segmentations of current patient are approved"

            # (4) gtvt waiting, gtvn approved (user can do nothing)
            elif (
                self.obs_study_gtvt_step == ObsStudyGTVtStep.WAIT_PRED
                and self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED
            ):
                text = "Neural Network is generating GTVt auto-segmentation, please wait..."

            # (5) gtvn waiting, gtvt approved (user can do nothing)
            elif (
                self.obs_study_gtvn_step == ObsStudyGTVnStep.WAIT_PRED
                and self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED
            ):
                text = "Neural Network is generating GTVn auto-segmentation, please wait..."

            # (6) gtvt being corrected, gtvn waiting
            # (7) gtvt being corrected, gtvn approved
            elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
                text = "Please correct the GTVt auto-segmentation"

            # (8) gtvn being corrected, gtvt waiting
            # (9) gtvn being corrected, gtvt approved
            elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
                text = "Please correct the GTVn auto-segmentation"

            # This block should never be reached
            else:
                g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        self._qimg_draw_text(
            qimg=qimg,
            text=text,
            pos=(left, top),
            color=self.color["green"],
        )

        # show delineated state on qimage
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
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
    def _display_score_on_top_left(self, qimg: QtGui.QImage):
        pass

    def _display_contour_name_on_bottom_left(self, qimg: QtGui.QImage):
        left = self._get_text_pos_left()
        bottom = self._get_text_pos_bottom(qimg)

        # user input text
        if (
            self.img_3d["gtvt.click"] is not None
            or self.img_3d["gtvt.delineation.{}".format(Plane.TRANSVERSE)] is not None
            or self.img_3d["gtvt.delineation.{}".format(Plane.CORONAL)] is not None
            or self.img_3d["gtvt.delineation.{}".format(Plane.SAGITTAL)] is not None
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
        self.__stop_obs_study_gtvt_process()
        self.__obs_study_gtvn_thread.stop()

        # update widgets
        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].setEnabled(True)
        if not self._collap["annotation"].isExpanded():
            self._collap["annotation"].expand()

        # clear data
        self._clear_img_3d()
        self.gtvt_click_pos_3d = None
        self.gtvn_clicks_pos_3d = List()

        # update current patient
        self._cur_patient = self.combox["patient"].currentText()

        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_ver()

        # load multi-modal imgs only, no labels
        self._load_multi_modal_imgs()

        # load idl gtvt/gtvn images
        self._load_idl_gtvt_data()
        self._load_idl_gtvn_data()

        # reset timers after cur_patient is updated
        self.__timer = Dict()
        for i in [
            ObsStudyTimer.PATIENT_TOTAL_TIME,
            ObsStudyTimer.CLICK_GTVT_CENTER,
            ObsStudyTimer.DELINEATE_GTVT,
            ObsStudyTimer.CLICK_GTVN_CENTERS,
            ObsStudyTimer.WAIT_GTVT_PRED,
            ObsStudyTimer.WAIT_GTVN_PRED,
            ObsStudyTimer.CORRECT_GTVT,
            ObsStudyTimer.CORRECT_GTVN,
        ]:
            self.__timer[i] = ObsStudyTimer(
                baseline_id=self._baseline_id,
                idl_gtvt_id=self._idl_id["gtvt"],
                patient=self._cur_patient,
                timer_name=i,
            )

        # load current obs study steps from json(after cur_patient is updated)
        # init and save obs study steps of all patients
        cur_patient_obs_study_step = g.load_json(self.__obs_study_step_json_path)[
            "patient={}".format(self._cur_patient)
        ]

        # load obs study gtvt/gtvn steps from json dict
        # new patient
        if cur_patient_obs_study_step == {}:
            self.obs_study_gtvt_step = ObsStudyGTVtStep.CLICK_CENTER
            self.obs_study_gtvn_step = ObsStudyGTVnStep.CLICK_CENTERS
        # previous patient
        else:
            self.obs_study_gtvt_step = cur_patient_obs_study_step["gtvt"]
            self.obs_study_gtvn_step = cur_patient_obs_study_step["gtvn"]

        # gtvt thread was interupted
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.WAIT_PRED:
            # revert back to the nearest step
            self.obs_study_gtvt_step = ObsStudyGTVtStep.DELINEATE

        # gtvn thread was interupted
        if self.obs_study_gtvn_step == ObsStudyGTVnStep.WAIT_PRED:
            # revert back to the nearest step
            self.obs_study_gtvn_step = ObsStudyGTVnStep.CLICK_CENTERS

        # call reset_cur_slice_id() after:
        # (1) _load_multi_modal_imgs
        # (2) _load_idl_gtvt_data(), will load gtvt_click_pos_3d
        # (3) _load_idl_gtvn_data(), will load gtvn_clicks_pos_3d
        # (4) obs study gtvt/gtvn steps are loaded
        self.reset_cur_slice_id()

        # start total timer
        self.__timer[ObsStudyTimer.PATIENT_TOTAL_TIME].end()
        self.__timer[ObsStudyTimer.PATIENT_TOTAL_TIME].start()

        # last step: goto current obs study steps
        if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
            self.__goto_click_gtvt_center()

        elif self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
            self.__goto_delineate_gtvt()

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            self.__goto_click_gtvn_centers()

        else:
            self.__goto_correct_pred()

    def ensure_slice_id_multiple(self, slice_id: int, slice_count: int):
        remainder = slice_id % self.interpolation_step
        slice_id -= remainder
        if (
            remainder > self.interpolation_step / 2
            and slice_id + self.interpolation_step <= slice_count - 1
        ):
            slice_id += self.interpolation_step
        slice_id = g.clamp_value(slice_id, (0, slice_count - 1))
        return slice_id

    def reset_cur_slice_id(self):
        if (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER
            or self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE
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

        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            if len(self.gtvn_clicks_pos_3d) == 0:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()
            else:
                self.cur_slice_id = self.__get_gtvn_center_slices_id()

        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED
            and self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED
        ):
            self.cur_slice_id = self.__get_gtvt_center_slices_id()

        elif (
            self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT
            or self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT
        ):
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()

            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                # sometimes there is not gtvn clicks, but there is always a gtvt click
                if len(self.gtvn_clicks_pos_3d) == 0:
                    self.cur_slice_id = self.__get_gtvt_center_slices_id()
                else:
                    self.cur_slice_id = self.__get_gtvn_center_slices_id()

    def __update_obs_study_step(
        self,
        obs_study_gtvt_step: str = None,
        obs_study_gtvn_step: str = None,
    ):
        if obs_study_gtvt_step is not None:
            self.obs_study_gtvt_step = obs_study_gtvt_step

        if obs_study_gtvn_step is not None:
            self.obs_study_gtvn_step = obs_study_gtvn_step

        # save updated obs study steps into json
        all_patients_obs_study_step = g.load_json(self.__obs_study_step_json_path)
        all_patients_obs_study_step["patient={}".format(self._cur_patient)][
            "gtvt"
        ] = self.obs_study_gtvt_step
        all_patients_obs_study_step["patient={}".format(self._cur_patient)][
            "gtvn"
        ] = self.obs_study_gtvn_step
        g.save_json(all_patients_obs_study_step, self.__obs_study_step_json_path)

        # # update idl time used
        # obs_study_step_list = [
        #     ObsStudyStep.CLICK_GTVT_CENTER,
        #     ObsStudyStep.DRAW_GTVT,
        #     ObsStudyStep.CLICK_GTVN_CENTER,
        #     ObsStudyStep.WAITING_GTVT,
        #     ObsStudyStep.WAITING_GTVN,
        #     ObsStudyStep.CORRECT_GTVT,
        #     ObsStudyStep.CORRECT_GTVN,
        # ]
        # idl_time_json_path = os.path.join(
        #     g.TRAIN_RESULTS_DIR,
        #     self._baseline_id,
        #     self._idl_id["gtvt"],  # only save it in gtvt folder
        #     "time_used.json",
        # )
        # time_log = g.load_json(idl_time_json_path)
        # time_log["patient={}".format(self._cur_patient)][self.obs_study_step] = duration
        # g.save_json(time_log, idl_time_json_path)

    def __refresh_todo_list(self):
        # (1) CLICK_GTVT_CENTER and DELINEATE_GTVT have the highest priority.
        if self.obs_study_gtvt_step in [
            ObsStudyGTVtStep.CLICK_CENTER,
            ObsStudyGTVtStep.DELINEATE,
        ]:
            completed_todo_labels = [TodoListLabel.SELECT_PATIENT]
            active_todo_labels = []
            not_start_todo_labels = [
                TodoListLabel.WAIT_GTVT_PRED,
                TodoListLabel.CORRECT_GTVT,
            ]

            # gtvt todo labels
            if self.obs_study_gtvt_step == ObsStudyGTVtStep.CLICK_CENTER:
                active_todo_labels.append(TodoListLabel.CLICK_GTVT_CENTER)
                not_start_todo_labels += [
                    TodoListLabel.DELINEATE_GTVT,
                    TodoListLabel.DELINEATE_GTVT_TRANSVERSE,
                    TodoListLabel.DELINEATE_GTVT_CORONAL,
                    TodoListLabel.DELINEATE_GTVT_SAGITTAL,
                ]
            elif self.obs_study_gtvt_step == ObsStudyGTVtStep.DELINEATE:
                completed_todo_labels.append(TodoListLabel.CLICK_GTVT_CENTER)
                active_todo_labels.append(TodoListLabel.DELINEATE_GTVT)
                # update the status of delineate gtvt sub todo labels
                self.__update_gtvt_delineated_status()
            else:
                g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

            # gtvn todo labels
            if self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
                not_start_todo_labels += [
                    TodoListLabel.CLICK_GTVN_CENTERS,  # wait until gtvt is clicked or delineated
                    TodoListLabel.WAIT_GTVN_PRED,
                    TodoListLabel.CORRECT_GTVN,
                ]
            elif self.obs_study_gtvn_step == ObsStudyGTVnStep.WAIT_PRED:
                completed_todo_labels.append(TodoListLabel.CLICK_GTVN_CENTERS)
                active_todo_labels.append(TodoListLabel.WAIT_GTVN_PRED)
                not_start_todo_labels.append(TodoListLabel.CORRECT_GTVN)

            elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
                completed_todo_labels += [
                    TodoListLabel.CLICK_GTVN_CENTERS,
                    TodoListLabel.WAIT_GTVN_PRED,
                ]
                # wait until gtvt is clicked or delineated
                not_start_todo_labels.append(TodoListLabel.CORRECT_GTVN)

            elif self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED:
                completed_todo_labels += [
                    TodoListLabel.CLICK_GTVN_CENTERS,
                    TodoListLabel.WAIT_GTVN_PRED,
                    TodoListLabel.CORRECT_GTVN,
                ]
            else:
                g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        # (2) Click GTVn centers has the 2nd highest priority.
        elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CLICK_CENTERS:
            # gtvt is clicked and delineated
            completed_todo_labels = [
                TodoListLabel.SELECT_PATIENT,
                TodoListLabel.CLICK_GTVT_CENTER,
                TodoListLabel.DELINEATE_GTVT,
                TodoListLabel.DELINEATE_GTVT_TRANSVERSE,
                TodoListLabel.DELINEATE_GTVT_CORONAL,
                TodoListLabel.DELINEATE_GTVT_SAGITTAL,
            ]
            # gtvn todo labels
            active_todo_labels = [TodoListLabel.CLICK_GTVN_CENTERS]
            not_start_todo_labels = [
                TodoListLabel.WAIT_GTVN_PRED,
                TodoListLabel.CORRECT_GTVN,
            ]
            # gtvt todo labels
            if self.obs_study_gtvt_step == ObsStudyGTVtStep.WAIT_PRED:
                active_todo_labels.append(TodoListLabel.WAIT_GTVT_PRED)
                not_start_todo_labels.append(TodoListLabel.CORRECT_GTVT)

            elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
                completed_todo_labels.append(TodoListLabel.WAIT_GTVT_PRED)
                # wait until gtvn centers are clicked
                not_start_todo_labels.append(TodoListLabel.CORRECT_GTVT)

            elif self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED:
                completed_todo_labels += [
                    TodoListLabel.WAIT_GTVT_PRED,
                    TodoListLabel.CORRECT_GTVT,
                ]
            else:
                g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        # (3) other conditions
        # obs_study_gtvt_step == WAIT_PRED / CORRECT / APPROVED, != CLICK_CENTER / DELINEATE
        # obs_study_gtvn_step == WAIT_PRED / CORRECT / APPROVED, != CLICK_CENTERS
        else:
            completed_todo_labels = [
                TodoListLabel.SELECT_PATIENT,
                TodoListLabel.CLICK_GTVT_CENTER,
                TodoListLabel.DELINEATE_GTVT,
                TodoListLabel.DELINEATE_GTVT_TRANSVERSE,
                TodoListLabel.DELINEATE_GTVT_CORONAL,
                TodoListLabel.DELINEATE_GTVT_SAGITTAL,
                TodoListLabel.CLICK_GTVN_CENTERS,
            ]
            active_todo_labels = []
            not_start_todo_labels = []

            # gtvt todo labels
            if self.obs_study_gtvt_step == ObsStudyGTVtStep.WAIT_PRED:
                active_todo_labels.append(TodoListLabel.WAIT_GTVT_PRED)
                not_start_todo_labels.append(TodoListLabel.CORRECT_GTVT)

            elif self.obs_study_gtvt_step == ObsStudyGTVtStep.CORRECT:
                completed_todo_labels.append(TodoListLabel.WAIT_GTVT_PRED)
                active_todo_labels.append(TodoListLabel.CORRECT_GTVT)

            elif self.obs_study_gtvt_step == ObsStudyGTVtStep.APPROVED:
                completed_todo_labels += [
                    TodoListLabel.WAIT_GTVT_PRED,
                    TodoListLabel.CORRECT_GTVT,
                ]
            else:
                g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

            # gtvn todo labels
            if self.obs_study_gtvn_step == ObsStudyGTVnStep.WAIT_PRED:
                active_todo_labels.append(TodoListLabel.WAIT_GTVN_PRED)
                not_start_todo_labels.append(TodoListLabel.CORRECT_GTVN)

            elif self.obs_study_gtvn_step == ObsStudyGTVnStep.CORRECT:
                completed_todo_labels.append(TodoListLabel.WAIT_GTVN_PRED)
                active_todo_labels.append(TodoListLabel.CORRECT_GTVN)

            elif self.obs_study_gtvn_step == ObsStudyGTVnStep.APPROVED:
                completed_todo_labels += [
                    TodoListLabel.WAIT_GTVN_PRED,
                    TodoListLabel.CORRECT_GTVN,
                ]
            else:
                g.error_exit(ErrMsg.OBS_STUDY_STEP_INVALID)

        for i in completed_todo_labels:
            self._text_label[i].set_status_completed()
        for i in active_todo_labels:
            self._text_label[i].set_status_active()
        for i in not_start_todo_labels:
            self._text_label[i].set_status_not_start()

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

        cur_round_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id[gtv],
            "patients",
            "patient={}".format(self._cur_patient),
            "round=01",
        )

        nii_name_list = ["pred", "correction", "correction.mask"]
        if gtv == "gtvt":
            nii_name_list.append("click")
        elif gtv == "gtvn":
            nii_name_list.append("clicks")

        for i in nii_name_list:
            nii_path = os.path.join(
                cur_round_dir,
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

        # load gtvt delineation
        if gtv == "gtvt":
            self._load_idl_gtvt_delineation(cur_round_dir)

    def __combine_pred_delineation_correction(self):
        if self.img_3d[Modal.CT] is None:
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

    def is_obs_study_window(self):
        return True

    def _switch_pen_eraser(self):
        if (
            self.obs_study_gtvt_step
            not in [ObsStudyGTVtStep.DELINEATE, ObsStudyGTVtStep.CORRECT]
            and self.obs_study_gtvn_step != ObsStudyGTVnStep.CORRECT
        ):
            return

        if self.drawing_mode == DrawingMode.GTVT_PEN:
            self.drawing_mode = DrawingMode.GTVT_ERASER
        elif self.drawing_mode == DrawingMode.GTVT_ERASER:
            self.drawing_mode = DrawingMode.GTVT_PEN
        elif self.drawing_mode == DrawingMode.GTVN_PEN:
            self.drawing_mode = DrawingMode.GTVN_ERASER
        elif self.drawing_mode == DrawingMode.GTVN_ERASER:
            self.drawing_mode = DrawingMode.GTVN_PEN
        else:
            pass

        self.refresh_mouse_cursor()

        # refresh img_frame to add or remove eraser circle
        img_frame_name = self._under_mouse_img_frame_name()
        if img_frame_name:
            self.img_frame[img_frame_name].update()
