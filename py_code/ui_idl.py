import os
import random

import cv2
import numpy as np
import qimage2ndarray
from custom import Debug, Dict, Dir, DrawingMode
from custom import Global as g
from custom import IDLStep, Img, Json, List, Modal, Nii, Plane, Time, Value
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
from PyQt5.QtWidgets import QMessageBox, QRadioButton
from scipy import ndimage
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
            if self._cur_slice_id == center_slice_id:
                self.paint_pos = event.pos()

            # if on other slices, switch to center slice
            else:
                self._cur_slice_id = center_slice_id
                self._refresh_rgb_imgs()
                self._refresh_title()

        elif idl_step == IDLStep.CORRECTION:
            self.paint_pos = event.pos()

    def draw_on_4_qlabels_move(self, event: QMouseEvent):
        if self.paint_pos is None:
            return

        pen_size = self.get_pen_size()
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
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
            painter.setPen(QPen(pen_color, pen_size, Qt.SolidLine, Qt.RoundCap))

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
        qimg = self.img_qlabel[Modal.CT].drawing_layer.toImage()
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
        if self._plane == Plane.SAGITTAL:
            actual_shape = self._3d_imgs[Modal.CT][:, :, 0].shape
        elif self._plane == Plane.CORONAL:
            actual_shape = self._3d_imgs[Modal.CT][:, 0, :].shape
        elif self._plane == Plane.TRANSVERSE:
            actual_shape = self._3d_imgs[Modal.CT][0, :, :].shape
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
            if self._plane == Plane.TRANSVERSE:
                segment = self._3d_imgs["gtvt.annotation"][t, :, :]
            elif self._plane == Plane.CORONAL:
                segment = self._3d_imgs["gtvt.annotation"][:, c, :]
            elif self._plane == Plane.SAGITTAL:
                segment = self._3d_imgs["gtvt.annotation"][:, :, s]

        elif idl_step == IDLStep.CORRECTION:
            t = c = s = self._cur_slice_id
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
                segment_type_list = ["correction", "annotation", "pred"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
                segment_type_list = ["correction", "pred"]
            # loop through correction->annotation->pred until finding un-empty slice
            for i in segment_type_list:
                _3d_img = self._3d_imgs["{}.{}".format(gtv, i)]
                if self._plane == Plane.TRANSVERSE:
                    segment = _3d_img[t, :, :].copy()
                elif self._plane == Plane.CORONAL:
                    segment = _3d_img[:, c, :].copy()
                elif self._plane == Plane.SAGITTAL:
                    segment = _3d_img[:, :, s].copy()

                if i != "pred":
                    kernel = np.ones((3, 3), np.uint8)
                    eroded_segment = cv2.erode(segment, kernel, iterations=1)
                    if eroded_segment.max() <= 0:
                        continue
                    else:
                        break
                else:
                    if segment.max() <= 0:
                        continue
                    else:
                        break

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

        # replace slice in 3d gtvt.annotation
        if idl_step == IDLStep.DRAW_GTVT:
            _3d_img = self._3d_imgs["gtvt.annotation"]
        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                _3d_img = self._3d_imgs["gtvt.correction"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                _3d_img = self._3d_imgs["gtvn.correction"]
        if self._plane == Plane.TRANSVERSE:
            _3d_img[t, :, :] = segment
        elif self._plane == Plane.CORONAL:
            _3d_img[:, c, :] = segment
        elif self._plane == Plane.SAGITTAL:
            _3d_img[:, :, s] = segment

        # save gtvt and gtvn corrections
        if idl_step == IDLStep.CORRECTION:
            self.__save_corrections()

        # update values
        self.paint_pos = None
        self.__update_gtvt_annotated_status()

        # update UI
        self.__clear_all_drawing_layers_on_4_qlabels()
        self._refresh_rgb_imgs()

    def __save_corrections(self):
        for gtv in ["gtvt", "gtvn"]:
            if self._3d_imgs["{}.correction".format(gtv)] is None:
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
            correction = self._3d_imgs["{}.correction".format(gtv)].copy()
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                correction = np.flip(correction, axis=2)
            # turn upside down
            correction = np.flip(correction, axis=0)
            # save
            Nii.save(
                img=correction,
                save_path=os.path.join(
                    cur_round_dir, "{}_correction.nii.gz".format(gtv)
                ),
                spacing=self._nii_spacing,
            )

    def __click_btn_pen(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.DRAW_GTVT:
            self.drawing_mode = DrawingMode.GTVT_PEN

        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVT_ERASER]:
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_ERASER]:
                self.drawing_mode = DrawingMode.GTVN_PEN

        if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECTION]:
            self.__set_mouse_cursor("pen")
            self._text_label["pen.size"].setText("Pen Size")

    def __click_btn_eraser(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.DRAW_GTVT:
            self.drawing_mode = DrawingMode.GTVT_ERASER

        elif idl_step == IDLStep.CORRECTION:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_ERASER]:
                self.drawing_mode = DrawingMode.GTVT_ERASER
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVT_ERASER]:
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
            # pos 0-transverse 1-coronal 2-saggital
            self._3d_imgs["gtvt.click"][pos[0]][pos[1]][pos[2]] = 1

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
            idl_gtvt_click = self._3d_imgs["gtvt.click"].copy()
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
            selected_slices[Plane.TRANSVERSE]["round=01"] = List(pos[0]).to_str()
            selected_slices[Plane.CORONAL]["round=01"] = List(pos[1]).to_str()
            selected_slices[Plane.SAGITTAL]["round=01"] = List(pos[2]).to_str()
            Json.save(
                data=selected_slices,
                path=os.path.join(cur_patient_dir, "selected_slices.json"),
            )

            # clean current step elements
            # DO NOT clear self.__gtvt_click_pos_3d, IDLStep.DRAW_GTVT will use it
            self.delete_all_crosses_on_4_qlabels()
            # new step
            self.set_cur_patient_idl_step(IDLStep.DRAW_GTVT)
            self.__save_idl_step()
            self._refresh_rgb_imgs()
            self._refresh_title()
            self._3d_imgs["gtvt.annotation"] = np.zeros_like(self._3d_imgs[Modal.CT])
            self.drawing_mode = DrawingMode.GTVT_PEN
            self.__set_mouse_cursor("pen")

        elif self.get_cur_patient_idl_step() == IDLStep.DRAW_GTVT:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                if self.__gtvt_annotated_status[plane] is False:
                    QMessageBox.information(
                        self,
                        "Information",
                        "Please draw GTVt in {} plane.".format(plane),
                        QMessageBox.Ok,
                    )
                    self._set_img_plane(new_plane=plane)
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
            gtvt_annotation_to_save = self._3d_imgs["gtvt.annotation"].copy()
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
            self._refresh_rgb_imgs()
            self._refresh_title()

        elif self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
            # add clicks into 3d img
            for pos in self.__gtvn_clicks_pos_3d:
                # pos 0-transverse 1-coronal 2-saggital
                self._3d_imgs["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1

            # clean current step elements
            self.delete_all_crosses_on_4_qlabels()

            # show gtvn center
            self._refresh_rgb_imgs()
            self._refresh_title()

            # copy data (dont change origin ndarray)
            idl_gtvn_clicks = self._3d_imgs["gtvn.clicks"].copy()
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
            self._refresh_rgb_imgs()
            self._refresh_title()
            self._3d_imgs["gtvt.correction"] = np.zeros_like(self._3d_imgs[Modal.CT])
            self._3d_imgs["gtvn.correction"] = np.zeros_like(self._3d_imgs[Modal.CT])
            self.drawing_mode = DrawingMode.GTVT_PEN
            self.__set_mouse_cursor("pen")

    # check annotation in 3 different planes
    def __update_gtvt_annotated_status(self) -> Dict:
        t, c, s = np.where(self._3d_imgs["gtvt.click"] == 1)
        t, c, s = int(t), int(c), int(s)
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if plane == Plane.TRANSVERSE:
                cur_plane_annotation = self._3d_imgs["gtvt.annotation"][t, :, :].copy()
                cur_plane_annotation[c, :] = 0
                cur_plane_annotation[:, s] = 0

            elif plane == Plane.CORONAL:
                cur_plane_annotation = self._3d_imgs["gtvt.annotation"][:, c, :].copy()
                cur_plane_annotation[t, :] = 0
                cur_plane_annotation[:, s] = 0

            elif plane == Plane.SAGITTAL:
                cur_plane_annotation = self._3d_imgs["gtvt.annotation"][:, :, s].copy()
                cur_plane_annotation[t, :] = 0
                cur_plane_annotation[:, c] = 0

            if cur_plane_annotation.max() == 0:
                self.__gtvt_annotated_status[plane] = False
            else:
                self.__gtvt_annotated_status[plane] = True

    def _refresh_rgb_imgs(self, replay_mode: bool = False):
        # no patient loaded
        if self._3d_imgs[Modal.CT] is None:
            # ask user to select a patient
            w = self.img_qlabel[Modal.CT].width()
            h = self.img_qlabel[Modal.CT].height()
            qimg = QImage(w, h, QImage.Format_RGB888)
            black = QColor(0, 0, 0)
            qimg.fill(black)
            self._add_msg_on_qimg(qimg)
            self.img_qlabel[Modal.CT].set_background(qimg)
            self.img_qlabel[Modal.CT].update()
            return

        super()._refresh_rgb_imgs(replay_mode)

    def _init_color(self):
        super()._init_color()
        # self._color["gtvt.annotation"] = self._color["yellow"]
        self._color["gtvt.correction"] = self._color["yellow"]
        self._color["gtvn.correction"] = self._color["cyan"]

    def __click_btn_clear(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.clear_gtvt_click_pos_3d()
            self.__refresh_crosses_on_4_qlabels()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.clear_gtvn_clicks_pos_3d()
            self.__refresh_crosses_on_4_qlabels()

        elif idl_step == IDLStep.DRAW_GTVT:
            self.__clear_all_drawing_layers_on_4_qlabels()

            # clear annotation on cur plane
            t, c, s = np.where(self._3d_imgs["gtvt.click"] == 1)
            t, c, s = int(t), int(c), int(s)
            # use mask to filter out the annotation on current anatomical plane
            if self._plane == Plane.TRANSVERSE:
                mask = np.zeros_like(self._3d_imgs["gtvt.annotation"][t, :, :])
                mask[c, :] = 1
                mask[:, s] = 1
                self._3d_imgs["gtvt.annotation"][t, :, :] *= mask
            elif self._plane == Plane.CORONAL:
                mask = np.zeros_like(self._3d_imgs["gtvt.annotation"][:, c, :])
                mask[t, :] = 1
                mask[:, s] = 1
                self._3d_imgs["gtvt.annotation"][:, c, :] *= mask
            elif self._plane == Plane.SAGITTAL:
                mask = np.zeros_like(self._3d_imgs["gtvt.annotation"][:, :, s])
                mask[t, :] = 1
                mask[:, c] = 1
                self._3d_imgs["gtvt.annotation"][:, :, s] *= mask

            # update gtvt annotated status
            self.__gtvt_annotated_status[self._plane] = False
            self._refresh_rgb_imgs()

        elif idl_step == IDLStep.CORRECTION:
            t = c = s = self._cur_slice_id
            for gtv in ["gtvt", "gtvn"]:
                _3d_img = self._3d_imgs["{}.correction".format(gtv)]
                if self._plane == Plane.TRANSVERSE:
                    _3d_img[t, :, :] = np.zeros_like(_3d_img[t, :, :])
                elif self._plane == Plane.CORONAL:
                    _3d_img[:, c, :] = np.zeros_like(_3d_img[:, c, :])
                elif self._plane == Plane.SAGITTAL:
                    _3d_img[:, :, s] = np.zeros_like(_3d_img[:, :, s])

            self.__save_corrections()
            self._refresh_rgb_imgs()

    def __get_gtvt_center_slice_id(self):
        if self.__gtvt_click_pos_3d is None:
            Debug.error_exit("no gtvt click")
        if self._plane == Plane.TRANSVERSE:
            center_slice_id = self.__gtvt_click_pos_3d[0]
        elif self._plane == Plane.CORONAL:
            center_slice_id = self.__gtvt_click_pos_3d[1]
        elif self._plane == Plane.SAGITTAL:
            center_slice_id = self.__gtvt_click_pos_3d[2]
        return center_slice_id

    def __get_gtvn_center_slice_id(self):
        if len(self.__gtvn_clicks_pos_3d) == 0:
            Debug.error_exit("no gtvn clicks")
        if self._plane == Plane.TRANSVERSE:
            center_slice_id = self.__gtvn_clicks_pos_3d[-1][0]
        elif self._plane == Plane.CORONAL:
            center_slice_id = self.__gtvn_clicks_pos_3d[-1][1]
        elif self._plane == Plane.SAGITTAL:
            center_slice_id = self.__gtvn_clicks_pos_3d[-1][2]
        return center_slice_id

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__refresh_crosses_on_4_qlabels()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.__refresh_crosses_on_4_qlabels()

    def _set_img_plane(
        self, connected_radio_btn: QRadioButton = None, new_plane: str = None
    ):
        super()._set_img_plane(
            connected_radio_btn=connected_radio_btn, new_plane=new_plane
        )
        self.__refresh_crosses_on_4_qlabels()

    def delete_all_crosses_on_4_qlabels(self):
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
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
                if self._plane == Plane.TRANSVERSE:
                    if self._cur_slice_id == d:
                        x = w / img_shape[2]
                        y = h / img_shape[1]

                elif self._plane == Plane.CORONAL:
                    if self._cur_slice_id == h:
                        x = w / img_shape[2]
                        y = d / img_shape[0]

                elif self._plane == Plane.SAGITTAL:
                    if self._cur_slice_id == w:
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
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].selected_cross.offset = pos

    def set_4_crosses_dragging_state(self, dragging: bool):
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].selected_cross.dragging = dragging

    def move_4_crosses(self, pos: QPoint):
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].selected_cross.move(pos)

    def delete_4_crosses(self):
        cross = self.img_qlabel[Modal.CT].selected_cross
        self.delete_click_pos(cross)
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].delete_selected_cross()

    # make this function public, CustomQLabel will use it
    def add_4_crosses(self, pos: QPoint, record_click_pos: bool):
        if self._3d_imgs[Modal.CT] is None:
            return

        # make sure new cross id is unique
        crosses_id_list = self.img_qlabel[Modal.CT].get_crosses_id_list()
        while 1:
            cross_id = random.randint(0, 2**16)
            if cross_id not in crosses_id_list:
                break
        # add crosses
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].add_cross(pos=pos, cross_id=cross_id)

        # add clicks into 3d img
        if record_click_pos:
            new_cross = self.img_qlabel[Modal.CT].get_cross_by_id(cross_id)
            self.add_click_pos(new_cross)

    def select_4_crosses(self, cross_id: int):
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].select_cross(cross_id)

    def get_rgb_img_roi(self):
        return self._rgb_img_roi

    def get_nii_spacing(self):
        return self._nii_spacing

    def get_img_plane(self):
        return self._plane

    def get_cur_slice(self):
        return self._cur_slice_id

    def get_3d_img_shape(self):
        if self._3d_imgs[Modal.CT] is not None:
            return self._3d_imgs[Modal.CT].shape
        else:
            return None

    def __init__(
        self,
        idl_remark: str = None,
        debug_mode: bool = False,
    ):
        # pass debug_mode parameter to the parent class
        super().__init__(idl_remark=idl_remark, debug_mode=debug_mode)

    def _init_ui_names(self):
        super()._init_ui_names()

        self._text_label["annotation.tools"] = self._text_label_annotation_tools
        self._text_label["idl.progress"] = self._text_label_idl_progress
        self._text_label["pen.size"] = self._text_label_pen_size

        self.__btn = Dict()
        self.__btn["pen"] = self._btn_pen
        self.__btn["eraser"] = self._btn_eraser
        self.__btn["clear"] = self._btn_clear
        self.__btn["confirm"] = self._btn_confirm

    def clear_gtvt_click_pos_3d(self):
        self.__gtvt_click_pos_3d = None

    def clear_gtvn_clicks_pos_3d(self):
        self.__gtvn_clicks_pos_3d = List()

    def _init_member_var(self, idl_remark: str = None, debug_mode: bool = False):
        super()._init_member_var()
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
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
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
        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            self.img_qlabel[i].drawing_layer = QPixmap(self.img_qlabel[i].size())
            self.img_qlabel[i].drawing_layer.fill(Qt.transparent)
            self.img_qlabel[i].update()

    def __set_mouse_cursor(self, cursor_type: str):
        cursor_size = 32  # no larger than 32
        cursor_pixmap = QPixmap(
            (os.path.join(g.PROJ_DIR, "icons", "{}_cursor.png".format(cursor_type)))
        )
        cursor_pixmap = cursor_pixmap.scaled(
            cursor_size, cursor_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if cursor_type == "pen":
            self.setCursor(QCursor(cursor_pixmap, 0, cursor_size * 0.95))
        else:
            self.setCursor(QCursor(cursor_pixmap, cursor_size * 0.2, cursor_size * 0.8))

    def _init_side_bar(self):
        super()._init_side_bar()

        # hide idl.gtvt/gtvn controls
        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._text_label[i].hide()
            self._combox[i].hide()
            self._arrow_btn["prev.{}".format(i)].hide()
            self._arrow_btn["next.{}".format(i)].hide()

        # show annotation controls
        # self._text_box_annotation_msg.show()
        self._progress_bar_idl.show()
        self._slider_pen_size.show()
        for i in ["annotation.tools", "idl.progress", "pen.size"]:
            self._text_label[i].show()
        for i in ["pen", "eraser", "clear", "confirm"]:
            self.__btn[i].show()

        # set text
        # self._text_box_annotation_msg.setText("Please Select a Patient")
        self._text_label["annotation.tools"].setText("Annotation Tools")
        self._text_label["idl.progress"].setText("Retraining Progress")

        # set fonts
        for i in ["annotation.tools", "idl.progress", "pen.size"]:
            self._text_label[i].setFont(self._font_bold)
        self._text_box_annotation_msg.setFont(self._font_bold)

        # set textbox read only
        self._text_box_annotation_msg.setReadOnly(True)

        # pen size slider
        self._slider_pen_size.setMinimum(1)
        self._slider_pen_size.setMaximum(11)
        self._slider_pen_size.setValue(6)

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

        # connect ui to functions
        # (put this at the end, because these functions will need the initialization above)
        self.__btn["pen"].clicked.connect(self.__click_btn_pen)
        self.__btn["eraser"].clicked.connect(self.__click_btn_eraser)
        self.__btn["clear"].clicked.connect(self.__click_btn_clear)
        self.__btn["confirm"].clicked.connect(self.__click_btn_confirm)

    def get_pen_size(self):
        return self._slider_pen_size.value()

    def _refresh_side_bar(self):
        (
            left,
            top,
            width,
            gap,
            text_height,
            bar_height,
            slider_height,
        ) = super()._refresh_side_bar(widgets_to_display=["patient"])

        annotation_msg_box_height = 80
        annotation_btn_width = 50
        annotation_btn_height = 40

        # annotation tools
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label["annotation.tools"].setGeometry(rect)
        self._text_label["annotation.tools"].show()
        top += text_height
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
        self._slider_pen_size.setGeometry(rect)
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
        self._clear_img_data()
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
                color=self._color["msg"],
            )
            return

        cur_patient_idl_step = self.get_cur_patient_idl_step()

        if cur_patient_idl_step == IDLStep.CLICK_GTVT_CENTER:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please click the center of the primary Gross Tumor Volumes (GTVt)",
                pos=(pos_x, pos_y),
                color=self._color["msg"],
            )

        elif cur_patient_idl_step == IDLStep.DRAW_GTVT:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please delineate the countour of GTVt in 3 different anatomical planes",
                pos=(pos_x, pos_y),
                color=self._color["msg"],
            )
            pos_y += 5
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                pos_y += 20
                text = Value.capitalized(plane)
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
                text="Please click the center of the malignant lymph nodes (GTVn)",
                pos=(pos_x, pos_y),
                color=self._color["msg"],
            )

        elif cur_patient_idl_step == IDLStep.CORRECTION:
            self._qimg_draw_text(
                qimg=qimg,
                text="Please correct the auto-segmentation",
                pos=(pos_x, pos_y),
                color=self._color["msg"],
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
            if self._3d_imgs["gtv{}.pred".format(i)] is not None:
                text = "GTV{}".format(i)
                self._qimg_draw_text(
                    qimg=qimg,
                    text=text,
                    pos=(pos_x, pos_y),
                    color=self._color["gtv{}.pred".format(i)],
                )
                pos_x += 45

    def _load_patient_data(self, idx: int = None):
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
        self._refresh_rgb_imgs()
        self._refresh_title()
        self.__refresh_crosses_on_4_qlabels()
        self.__save_idl_step()

    def _reset_cur_slice_id(self):
        if (
            self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER
            or self.get_cur_patient_idl_step() == IDLStep.DRAW_GTVT
        ):
            if self.__gtvt_click_pos_3d is None:
                self._cur_slice_id = self._get_middle_slice_id()
            else:
                self._cur_slice_id = self.__get_gtvt_center_slice_id()

        elif (
            self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER
            or self.get_cur_patient_idl_step() == IDLStep.CORRECTION
        ):
            if len(self.__gtvn_clicks_pos_3d) == 0:
                self._cur_slice_id = self.__get_gtvt_center_slice_id()
            else:
                self._cur_slice_id = self.__get_gtvn_center_slice_id()

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
        patient_dir = self._load_idl_gtv_data(gtv="gtvt")

        # load gtvt click and annotation
        for i in ["click", "annotation"]:
            nii_path = os.path.join(patient_dir, "round=01", "gtvt_{}.nii.gz".format(i))
            if os.path.exists(nii_path):
                self._3d_imgs["gtvt.{}".format(i)] = self._load_3d_img(
                    path=nii_path, binary=True
                )
            else:
                self._3d_imgs["gtvt.{}".format(i)] = np.zeros(
                    self._3d_imgs[Modal.CT].shape, dtype=np.float32
                )

    def _load_idl_gtvn_data(self):
        patient_dir = self._load_idl_gtv_data(gtv="gtvn")
        # load gtvn click
        gtvn_clicks_nii_path = os.path.join(
            patient_dir, "round=01", "gtvn_clicks.nii.gz"
        )
        if os.path.exists(gtvn_clicks_nii_path):
            self._3d_imgs["gtvn.clicks"] = self._load_3d_img(
                path=gtvn_clicks_nii_path, binary=True
            )
        else:
            self._3d_imgs["gtvn.clicks"] = np.zeros(
                self._3d_imgs[Modal.CT].shape, dtype=np.float32
            )

    def _load_idl_gtv_data(self, gtv: str) -> str:
        patient_dir = os.path.join(
            g.TRAIN_RESULTS_DIR,
            self._baseline_id,
            self._idl_id[gtv],
            "patients",
            "patient={}".format(self._cur_patient),
        )

        # current patient dir exists
        if os.path.exists(patient_dir):
            round_dirs = Dir.get_sub_dirs(
                patient_dir, key_word="round=", full_path=True
            )
            # choose the last round
            if len(round_dirs) > 0:
                round_dir = round_dirs[-1]
                pred_path = os.path.join(round_dir, "{}_pred.nii.gz".format(gtv))

                # find idl pred, load it
                if os.path.exists(pred_path):
                    self._3d_imgs["{}.pred".format(gtv)] = Img.binarize(
                        self._load_3d_img(pred_path)
                    )
                # cant find idl pred, clear 3d img
                else:
                    self._3d_imgs["{}.pred".format(gtv)] = None

            # no round dirs found
            else:
                self._3d_imgs["{}.pred".format(gtv)] = None

        # cant find cur patient dir
        else:
            # Dir.create(patient_dir)
            self._3d_imgs["{}.pred".format(gtv)] = None

        return patient_dir
