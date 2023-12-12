import os

import cv2
import numpy as np
import qimage2ndarray
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Timer, Value
from numpy import ndarray
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPoint, QSize, Qt, QThread, pyqtSignal
from scipy import ndimage
from str_lib import DisplayMode, DrawingMode, IDLStep, Modal, Plane
from superqt import QCollapsible
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt
from ui_custom_qlabel import CustomQLabel
from ui_draggable_cross import DraggableCross
from ui_replay import UiReplay


class IDLGTVnThread(QThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_completed = False

    def set_param(
        self,
        idl_gtvn_id: str,
        patient: str,
        idl_gtvn_clicks: ndarray,
        dataset_part: str,
        dataset_ver: str,
        debug_mode: bool,
    ):
        self.__idl_gtvn_id = idl_gtvn_id
        self.__patient = patient
        self.__idl_gtvn_clicks = idl_gtvn_clicks
        self.__dataset_part = dataset_part
        self.__dataset_ver = dataset_ver
        self.__debug_mode = debug_mode
        self.is_completed = False

    def run(self):
        training_idl_gtvn = TrainingIDLGTVn(self.progress_signal)
        training_idl_gtvn.real_idl(
            idl_gtvn_id=self.__idl_gtvn_id,
            patient=self.__patient,
            idl_gtvn_clicks=self.__idl_gtvn_clicks,
            dataset_part=self.__dataset_part,
            dataset_ver=self.__dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self.is_completed = True
        self.complete_signal.emit()


class IDLGTVtThread(QThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_completed = False

    def set_param(
        self,
        idl_gtvt_id: str,
        patient: str,
        dataset_ver: str,
        debug_mode: bool,
    ):
        self.__idl_gtvt_id = idl_gtvt_id
        self.__patient = patient
        self.__dataset_ver = dataset_ver
        self.__debug_mode = debug_mode
        self.is_completed = False

    def run(self):
        training_idl_gtvt = TrainingIDLGTVt(self.progress_signal)
        training_idl_gtvt.real_idl(
            idl_gtvt_id=self.__idl_gtvt_id,
            patient=self.__patient,
            dataset_ver=self.__dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self.is_completed = True
        self.complete_signal.emit()


class UiIDL(UiReplay):
    def draw_on_img_qlabels_press(self, event: QtGui.QMouseEvent):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step not in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            return

        if idl_step == IDLStep.DRAW_GTVT:
            gtvt_center_slice_id = self.__get_gtvt_center_slices_id()
            # (1) if on center slice, start painting
            if self.cur_slice_id == gtvt_center_slice_id:
                self.paint_pos = event.pos()
            # (2) if on other slices, switch to center slice
            else:
                self.cur_slice_id = gtvt_center_slice_id
                self.refresh_img_qlabels()

        elif idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self.paint_pos = event.pos()

    def draw_on_img_qlabels_move(
        self, event: QtGui.QMouseEvent, img_qlabel: CustomQLabel
    ):
        if self.paint_pos is None:
            return

        pen_size = self.get_pen_size()
        eraser_size = pen_size + 2
        eraser_color = QtGui.QColor(*self._color["eraser"])

        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [img_qlabel.plane]
        for i in img_name_list:
            painter = QtGui.QPainter(self.img_qlabel[i].drawing_layer)

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

            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                pen_color = QtGui.QColor(*self._color["gtvt.pred"])
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                pen_color = QtGui.QColor(*self._color["gtvn.pred"])

            if self.drawing_mode in [DrawingMode.GTVT_ERASER, DrawingMode.GTVN_ERASER]:
                painter.setPen(
                    QtGui.QPen(eraser_color, eraser_size, Qt.SolidLine, Qt.RoundCap)
                )
                self.img_qlabel[i].pen_mode = False
            elif self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
                painter.setPen(
                    QtGui.QPen(pen_color, pen_size, Qt.SolidLine, Qt.RoundCap)
                )
                self.img_qlabel[i].pen_mode = True

            painter.drawLine(self.paint_pos, event.pos())

            self.img_qlabel[i].update()  # schedule a repaint

        self.paint_pos = event.pos()  # update paint pos

    def draw_on_img_qlabels_release(self, img_qlabel: CustomQLabel):
        if self.paint_pos is None:
            return

        # binarize threshold
        # this is for saving qimage as ndarray
        # binarization is needed before and after resize the ndarray
        binary_threshold = 0.5

        # save drawing layer into 2d ndarray
        # qpixmap to a qimage
        qimg = img_qlabel.drawing_layer.toImage()
        # qimage to ndarray
        annotation_2d = qimage2ndarray.alpha_view(qimg).astype(np.float32)
        annotation_2d /= 255

        # binarization (before resize)
        annotation_2d = Img.binarize(img=annotation_2d, threshold=binary_threshold)

        # crop annotation_2d based on roi
        x = img_qlabel.roi.x
        y = img_qlabel.roi.y
        width = img_qlabel.roi.width
        height = img_qlabel.roi.height
        annotation_2d = annotation_2d[y : y + height, x : x + width]

        # resize to actual size
        if img_qlabel.plane == Plane.SAGITTAL:
            actual_shape = self.img_3d[Modal.CT][:, :, 0].shape
        elif img_qlabel.plane == Plane.CORONAL:
            actual_shape = self.img_3d[Modal.CT][:, 0, :].shape
        elif img_qlabel.plane == Plane.TRANSVERSE:
            actual_shape = self.img_3d[Modal.CT][0, :, :].shape
        annotation_2d = cv2.resize(
            annotation_2d,
            (actual_shape[1], actual_shape[0]),
            interpolation=cv2.INTER_AREA,  # best for scaling down
        )

        # binarization (after resize)
        annotation_2d = Img.binarize(img=annotation_2d, threshold=binary_threshold)

        # add 2d annotation on 3d annotation ndarray
        idl_step = self.get_cur_patient_idl_step()
        # (1)gtvt annotation
        if idl_step == IDLStep.DRAW_GTVT:
            t, c, s = self.gtvt_click_pos_3d
            if img_qlabel.plane == Plane.TRANSVERSE:
                segment = self.img_3d["gtvt.annotation"][t, :, :]
            elif img_qlabel.plane == Plane.CORONAL:
                segment = self.img_3d["gtvt.annotation"][:, c, :]
            elif img_qlabel.plane == Plane.SAGITTAL:
                segment = self.img_3d["gtvt.annotation"][:, :, s]
        # (2)correction
        elif idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            t = c = s = self.cur_slice_id[img_qlabel.plane]
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
            _3d_img = self.img_3d["{}.pred.final".format(gtv)]
            if img_qlabel.plane == Plane.TRANSVERSE:
                segment = _3d_img[t, :, :].copy()
            elif img_qlabel.plane == Plane.CORONAL:
                segment = _3d_img[:, c, :].copy()
            elif img_qlabel.plane == Plane.SAGITTAL:
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
        elif idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                _3d_img = self.img_3d["gtvt.correction"]
                _3d_mask = self.img_3d["gtvt.correction.mask"]
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                _3d_img = self.img_3d["gtvn.correction"]
                _3d_mask = self.img_3d["gtvn.correction.mask"]

        # replace slice
        if img_qlabel.plane == Plane.TRANSVERSE:
            _3d_img[t, :, :] = segment
        elif img_qlabel.plane == Plane.CORONAL:
            _3d_img[:, c, :] = segment
        elif img_qlabel.plane == Plane.SAGITTAL:
            _3d_img[:, :, s] = segment

        # update masks then save corrections and masks
        if idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            # (1)update correction masks
            if img_qlabel.plane == Plane.TRANSVERSE:
                if segment.max() == 0:
                    _3d_mask[t, :, :] = np.zeros_like(segment)
                else:
                    _3d_mask[t, :, :] = np.ones_like(segment)
            elif img_qlabel.plane == Plane.CORONAL:
                if segment.max() == 0:
                    _3d_mask[:, c, :] = np.zeros_like(segment)
                else:
                    _3d_mask[:, c, :] = np.ones_like(segment)
            elif img_qlabel.plane == Plane.SAGITTAL:
                if segment.max() == 0:
                    _3d_mask[:, :, s] = np.zeros_like(segment)
                else:
                    _3d_mask[:, :, s] = np.ones_like(segment)
            # (2)save corrections and masks
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
            self.__save_corrections_and_masks(gtv)

        # update values
        self.paint_pos = None
        self.__update_gtvt_annotated_status()
        self.__combine_pred_annotation_correction()

        # update UI
        self.__clear_all_drawing_layers(img_qlabel)
        self.refresh_img_qlabels()

    def __save_corrections_and_masks(self, gtv: str):
        if gtv not in ["gtvt", "gtvn"]:
            Debug.error_exit("Value of 'gtv' must be one of 'gtvt' or 'gtvn'!")

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

    # this function is connected to widget, dont set input params to this function
    def __on_btn_pen_clicked(self):
        idl_step = self.get_cur_patient_idl_step()

        # (1) update drawing mode
        if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECT_GTVT]:
            self.drawing_mode = DrawingMode.GTVT_PEN
        elif idl_step == IDLStep.CORRECT_GTVN:
            self.drawing_mode = DrawingMode.GTVN_PEN
        elif idl_step == IDLStep.CORRECT_BOTH:
            if self.drawing_mode == DrawingMode.GTVT_ERASER:
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self.drawing_mode == DrawingMode.GTVN_ERASER:
                self.drawing_mode = DrawingMode.GTVN_PEN

        # (2) update widgets
        if idl_step in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self.__set_mouse_cursor("pen")
            self._text_label["draw.size"].setText("Pen Size")

    # this function is connected to widget, dont set input params to this function
    def __on_btn_eraser_clicked(self):
        idl_step = self.get_cur_patient_idl_step()

        # (1) update drawing mode
        if idl_step in [IDLStep.DRAW_GTVT, IDLStep.CORRECT_GTVT]:
            self.drawing_mode = DrawingMode.GTVT_ERASER
        elif idl_step == IDLStep.CORRECT_GTVN:
            self.drawing_mode = DrawingMode.GTVN_ERASER
        elif idl_step == IDLStep.CORRECT_BOTH:
            if self.drawing_mode == DrawingMode.GTVT_PEN:
                self.drawing_mode = DrawingMode.GTVT_ERASER
            elif self.drawing_mode == DrawingMode.GTVN_PEN:
                self.drawing_mode = DrawingMode.GTVN_ERASER

        # (2) update widgets
        if idl_step in [
            IDLStep.DRAW_GTVT,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self.__set_mouse_cursor("eraser")
            self._text_label["draw.size"].setText("Eraser Size")

    def __confirm_gtvt_center(self):
        # (1) check if there is gtvt click
        if self.gtvt_click_pos_3d is None:
            QtWidgets.QMessageBox.information(
                self,
                "Information",
                "GTVt center not detected.",
                QtWidgets.QMessageBox.Ok,
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

        # (5) clean cross and refresh imgs
        self.delete_all_crosses()
        # goto next step
        self.update_cur_patient_idl_step(IDLStep.DRAW_GTVT)
        self.__idl_gtvt_thread.is_completed = False
        self.__idl_gtvn_thread.is_completed = False
        # refresh to show gtvt click
        self.refresh_img_qlabels()
        self.img_3d["gtvt.annotation"] = np.zeros_like(self.img_3d[Modal.CT])
        self.drawing_mode = DrawingMode.GTVT_PEN

        # (6) update widgets
        self.__set_mouse_cursor("pen")
        # enable pen and eraser for drawing gtvt
        for i in ["pen", "eraser"]:
            self.__btn[i].setEnabled(True)
        self._slider["draw.size"].show()
        self._text_label["draw.size"].show()

    def __confirm_gtvt_annotation(self):
        # (1) check gtvt annotated statues
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if self.__gtvt_annotated_status[plane] is False:
                QtWidgets.QMessageBox.information(
                    self,
                    "Information",
                    "Please draw GTVt in {} plane.".format(plane),
                    QtWidgets.QMessageBox.Ok,
                )
                self._modal_fixed_mode_switch_plane(plane)
                if self.drawing_mode == DrawingMode.GTVT_ERASER:
                    self.drawing_mode = DrawingMode.GTVT_PEN
                    self.__set_mouse_cursor("pen")
                    self._text_label["draw.size"].setText("Pen Size")
                return

        # (2) save gtvt annotation
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

        # (3) update widgets and refresh img qlabel before start thread
        # (for better user experience)
        # update idl step here because refresh_img_qlabels() will need it
        self.update_cur_patient_idl_step(IDLStep.CLICK_GTVN_CENTER)
        self.__idl_gtvn_thread.is_completed = False
        # restore default cursor
        self.setCursor(Qt.ArrowCursor)
        for i in ["pen", "eraser"]:
            self.__btn[i].setEnabled(False)
        self._slider["draw.size"].hide()
        self._text_label["draw.size"].hide()
        # refresh msg on qlabel (only refresh the top-left img_qlabel)
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            self.refresh_img_qlabels(Modal.CT)
        else:
            self.refresh_img_qlabels(Plane.TRANSVERSE)

        # (4)start real idl gtvt
        self._text_label["gtvt.progress"].show()
        self.__progress_bar["gtvt"].show()
        self.__idl_gtvt_thread.set_param(
            idl_gtvt_id=self._idl_id["gtvt"],
            patient=self._cur_patient,
            dataset_ver=self._dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self.__idl_gtvt_thread.start()

    def __update_idl_gtvt_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        Value.limit_range(progress_int, (0, 100))
        self.__progress_bar["gtvt"].setValue(progress_int)

    def __on_idl_gtvt_thread_finished(self):
        self._text_label["gtvt.progress"].hide()
        self.__progress_bar["gtvt"].hide()

        # update idl step and widgets
        # (1) idl.gtvn is completed
        if self.__idl_gtvn_thread.is_completed:
            self.update_cur_patient_idl_step(IDLStep.CORRECT_BOTH)
            # show radio buttons
            for i in ["gtvt", "gtvn"]:
                self._radio_btn["correct.{}".format(i)].show()
            # dont change drawing mode and radio buttons, because user is correcting gtvn

        # (2) idl.gtvn thread is not completed, only correct gtvt
        else:
            # do nothing if user has not submitted gtvn clicks
            if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
                pass

            # if idl step is "waiting", show correction tools for gtvt pred
            elif self.get_cur_patient_idl_step() == IDLStep.WAITING:
                self.update_cur_patient_idl_step(IDLStep.CORRECT_GTVT)
                self._radio_btn["correct.gtvt"].setChecked(True)
                self.__set_mouse_cursor("pen")
                self.drawing_mode = DrawingMode.GTVT_PEN
                for i in ["pen", "eraser", "clear"]:
                    self.__btn[i].setEnabled(True)
                self._slider["draw.size"].show()
                self._text_label["draw.size"].show()

            # idl step can not be other values
            else:
                Debug.error_exit("idl step error")

        # update gtvt 3d imgs and qlabels
        self._load_idl_gtvt_data()
        self.__combine_pred_annotation_correction()
        self.refresh_img_qlabels()
        # init correction and mask
        for i in ["gtvt.correction", "gtvt.correction.mask"]:
            self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])

    def __confirm_gtvn_center(self):
        # (1)add clicks into 3d img
        if self.img_3d["gtvn.clicks"] is None:
            self.img_3d["gtvn.clicks"] = np.zeros_like(self.img_3d[Modal.CT])
        for pos in self.gtvn_clicks_pos_3d:
            # pos 0-transverse 1-coronal 2-saggital
            self.img_3d["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1

        # (2) update widget before idl.gtvn thread (for better user experience)
        # gtvn thread is still running, only correct gtvt
        if self.__idl_gtvt_thread.is_completed:
            # update idl step before refresh img qlabels
            self.update_cur_patient_idl_step(IDLStep.CORRECT_GTVT)
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.__set_mouse_cursor("pen")
            self.drawing_mode = DrawingMode.GTVT_PEN
            for i in ["pen", "eraser", "clear"]:
                self.__btn[i].setEnabled(True)
            self._slider["draw.size"].show()
            self._text_label["draw.size"].show()

        # gtvt thread is not completedgoto waiting step
        else:
            # update idl step before refresh img qlabels
            self.update_cur_patient_idl_step(IDLStep.WAITING)
            self.__btn["clear"].setEnabled(False)

        # disable "confirm" button, its not needed anymore
        self.__btn["confirm"].setEnabled(False)

        #  delete cross and refresh to show gtvn clicks
        self.delete_all_crosses()
        self.refresh_img_qlabels()

        # (3) transform gtvn clicks for idl.gtvn thread
        # copy data (dont change origin ndarray)
        idl_gtvn_clicks = self.img_3d["gtvn.clicks"].copy()
        # flip left/right for 1mm data
        if self._nii_spacing[2] == 1.0:
            idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=2)
        # turn upside down
        idl_gtvn_clicks = np.flip(idl_gtvn_clicks, axis=0)

        # (4) start real idl gtvn
        self._text_label["gtvn.progress"].show()
        self.__progress_bar["gtvn"].show()
        self.__idl_gtvn_thread.set_param(
            idl_gtvn_id=self._idl_id["gtvn"],
            patient=self._cur_patient,
            idl_gtvn_clicks=idl_gtvn_clicks,
            dataset_part=self._dataset_part,
            dataset_ver=self._dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self.__idl_gtvn_thread.start()

    def __update_idl_gtvn_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        Value.limit_range(progress_int, (0, 100))
        self.__progress_bar["gtvn"].setValue(progress_int)

    def __on_idl_gtvn_thread_finished(self):
        # update widgets
        self._text_label["gtvn.progress"].hide()
        self.__progress_bar["gtvn"].hide()

        # update idl step and widgets
        # (1) gtvt thread is completed
        if self.__idl_gtvt_thread.is_completed:
            self.update_cur_patient_idl_step(IDLStep.CORRECT_BOTH)
            # show radio buttons
            for i in ["gtvt", "gtvn"]:
                self._radio_btn["correct.{}".format(i)].show()
            # dont change drawing mode, because user is correcting gtvt

        # (2) only correct gtvn
        else:
            self.update_cur_patient_idl_step(IDLStep.CORRECT_GTVN)
            self._radio_btn["correct.gtvn"].setChecked(True)
            self.__set_mouse_cursor("pen")
            self.drawing_mode = DrawingMode.GTVN_PEN
            for i in ["pen", "eraser", "clear"]:
                self.__btn[i].setEnabled(True)
            self._slider["draw.size"].show()
            self._text_label["draw.size"].show()

        # show gtvt/gtvn switch radio buttons
        if self.get_cur_patient_idl_step() == IDLStep.CORRECT_BOTH:
            for i in ["gtvt", "gtvn"]:
                self._radio_btn["correct.{}".format(i)].show()

        # update 3d imgs and qlabels
        self._load_idl_gtvn_data()
        self.__combine_pred_annotation_correction()
        self.refresh_img_qlabels()
        # init correction and mask
        for i in ["gtvn.correction", "gtvn.correction.mask"]:
            self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])

    # this function is connected to widget, dont set input params to this function
    def __on_btn_confirm_clicked(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.__confirm_gtvt_center()
        elif idl_step == IDLStep.DRAW_GTVT:
            self.__confirm_gtvt_annotation()
        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.__confirm_gtvn_center()

    # check annotation in 3 different planes
    def __update_gtvt_annotated_status(self) -> Dict:
        t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
        t, c, s = int(t), int(c), int(s)
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if plane == Plane.TRANSVERSE:
                cur_plane_annotation = self.img_3d["gtvt.annotation"][t, :, :].copy()
                cur_plane_annotation[c, :] = 0
                cur_plane_annotation[:, s] = 0

            elif plane == Plane.CORONAL:
                cur_plane_annotation = self.img_3d["gtvt.annotation"][:, c, :].copy()
                cur_plane_annotation[t, :] = 0
                cur_plane_annotation[:, s] = 0

            elif plane == Plane.SAGITTAL:
                cur_plane_annotation = self.img_3d["gtvt.annotation"][:, :, s].copy()
                cur_plane_annotation[t, :] = 0
                cur_plane_annotation[:, c] = 0

            if cur_plane_annotation.max() == 0:
                self.__gtvt_annotated_status[plane] = False
            else:
                self.__gtvt_annotated_status[plane] = True

    def refresh_img_qlabels(self, img_name=None):
        # no patient loaded
        if self.img_3d[Modal.CT] is None:
            if self.display_mode() == DisplayMode.PLANE_FIXED:
                img_name = Plane.TRANSVERSE
            else:
                img_name = Modal.CT
            # ask user to select a patient
            w = self.img_qlabel[img_name].width()
            h = self.img_qlabel[img_name].height()
            qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB888)
            black = QtGui.QColor(0, 0, 0)
            qimg.fill(black)
            self._add_msg_on_qimg(qimg)
            self.img_qlabel[img_name].set_background(qimg)
            self.img_qlabel[img_name].update()
            return

        super().refresh_img_qlabels(replay_mode=False, img_name=img_name)

    def __set_mouse_cursor(self, cursor_type: str):
        if cursor_type not in ["pen", "eraser"]:
            Debug.error_exit("'cursor_type' must be one of 'pen' or 'eraser'!")

        cursor_size = 32  # no larger than 32
        cursor_pixmap = QtGui.QPixmap(
            (os.path.join(g.PROJ_DIR, "icons", "{}_cursor.png".format(cursor_type)))
        )
        cursor_pixmap = cursor_pixmap.scaled(
            cursor_size, cursor_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if cursor_type == "pen":
            self.setCursor(QtGui.QCursor(cursor_pixmap, 0, cursor_size * 0.95))
        elif cursor_type == "eraser":
            self.setCursor(
                QtGui.QCursor(cursor_pixmap, cursor_size * 0.2, cursor_size * 0.8)
            )

    def _init_color(self):
        super()._init_color()
        self._color["gtvt.annotation"] = self._color["yellow"]
        self._color["gtvt.correction"] = self._color["yellow"]
        self._color["gtvn.correction"] = self._color["cyan"]
        self._color["eraser"] = self._color["black"]
        self._color["gtvt.pred.final"] = self._color["gtvt.pred"]
        self._color["gtvn.pred.final"] = self._color["gtvn.pred"]

    # this function is connected to widget, dont set input params to this function
    def __on_btn_clear_clicked(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.clear_gtvt_click_pos_3d()
            self.refresh_crosses_on_qlabels()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.clear_gtvn_clicks_pos_3d()
            self.refresh_crosses_on_qlabels()

        elif idl_step == IDLStep.DRAW_GTVT:
            # modality fixed mode: clear annotation on cur plane
            if self.display_mode() == DisplayMode.MODAL_FIXED:
                t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
                t, c, s = int(t), int(c), int(s)
                # use mask to filter out the annotation on current anatomical plane
                if self.img_qlabel[Modal.CT].plane == Plane.TRANSVERSE:
                    mask = np.zeros_like(self.img_3d["gtvt.annotation"][t, :, :])
                    mask[c, :] = 1
                    mask[:, s] = 1
                    self.img_3d["gtvt.annotation"][t, :, :] *= mask
                elif self.img_qlabel[Modal.CT].plane == Plane.CORONAL:
                    mask = np.zeros_like(self.img_3d["gtvt.annotation"][:, c, :])
                    mask[t, :] = 1
                    mask[:, s] = 1
                    self.img_3d["gtvt.annotation"][:, c, :] *= mask
                elif self.img_qlabel[Modal.CT].plane == Plane.SAGITTAL:
                    mask = np.zeros_like(self.img_3d["gtvt.annotation"][:, :, s])
                    mask[t, :] = 1
                    mask[:, c] = 1
                    self.img_3d["gtvt.annotation"][:, :, s] *= mask
                # update gtvt annotated status
                self.__gtvt_annotated_status[self.img_qlabel[Modal.CT].plane] = False

            # plane fixed mode: clear whole annotation
            else:
                self.img_3d["gtvt.annotation"] = np.zeros_like(self.img_3d[Modal.CT])
                # update gtvt annotated status
                for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                    self.__gtvt_annotated_status[i] = False

            self.__combine_pred_annotation_correction()
            self.refresh_img_qlabels()

        elif idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"

            # modality fixed mode: clear correction on cur plane
            if self.display_mode() == DisplayMode.MODAL_FIXED:
                # use ct img qlabel's plane
                t = c = s = self.cur_slice_id[self.img_qlabel[Modal.CT].plane]
                _3d_img = self.img_3d["{}.correction".format(gtv)]
                _3d_mask = self.img_3d["{}.correction.mask".format(gtv)]
                if self.img_qlabel[Modal.CT].plane == Plane.TRANSVERSE:
                    _3d_img[t, :, :] = np.zeros_like(_3d_img[t, :, :])
                    _3d_mask[t, :, :] = np.zeros_like(_3d_mask[t, :, :])
                elif self.img_qlabel[Modal.CT].plane == Plane.CORONAL:
                    _3d_img[:, c, :] = np.zeros_like(_3d_img[:, c, :])
                    _3d_mask[:, c, :] = np.zeros_like(_3d_mask[:, c, :])
                elif self.img_qlabel[Modal.CT].plane == Plane.SAGITTAL:
                    _3d_img[:, :, s] = np.zeros_like(_3d_img[:, :, s])
                    _3d_mask[:, :, s] = np.zeros_like(_3d_mask[:, :, s])

            # plane fixed mode: clear whole correction
            else:
                self.img_3d["{}.correction".format(gtv)] = np.zeros_like(
                    self.img_3d[Modal.CT]
                )
                self.img_3d["{}.correction.mask".format(gtv)] = np.zeros_like(
                    self.img_3d[Modal.CT]
                )

            self.__save_corrections_and_masks(gtv)
            self.__combine_pred_annotation_correction()
            self.refresh_img_qlabels()

    def __get_gtvt_center_slices_id(self):
        center_slices_id = Dict()
        if self.gtvt_click_pos_3d is None:
            center_slices_id[Plane.TRANSVERSE] = None
            center_slices_id[Plane.CORONAL] = None
            center_slices_id[Plane.SAGITTAL] = None
        else:
            center_slices_id[Plane.TRANSVERSE] = self.gtvt_click_pos_3d[0]
            center_slices_id[Plane.CORONAL] = self.gtvt_click_pos_3d[1]
            center_slices_id[Plane.SAGITTAL] = self.gtvt_click_pos_3d[2]
        return center_slices_id

    def __get_gtvn_center_slices_id(self):
        center_slices_id = Dict()
        if len(self.gtvn_clicks_pos_3d) == 0:
            center_slices_id[Plane.TRANSVERSE] = None
            center_slices_id[Plane.CORONAL] = None
            center_slices_id[Plane.SAGITTAL] = None
        else:
            center_slices_id[Plane.TRANSVERSE] = self.gtvn_clicks_pos_3d[-1][0]
            center_slices_id[Plane.CORONAL] = self.gtvn_clicks_pos_3d[-1][1]
            center_slices_id[Plane.SAGITTAL] = self.gtvn_clicks_pos_3d[-1][2]
        return center_slices_id

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_crosses_on_qlabels()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.refresh_crosses_on_qlabels()

    def _modal_fixed_mode_switch_plane(self, new_plane: str = None):
        super()._modal_fixed_mode_switch_plane(new_plane)
        self.refresh_crosses_on_qlabels()

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
            self.img_qlabel[i].delete_all_crosses()

    def refresh_crosses_on_qlabels(self, img_name: str = None):
        if self.get_cur_patient_idl_step() not in [
            IDLStep.CLICK_GTVT_CENTER,
            IDLStep.CLICK_GTVN_CENTER,
        ]:
            return

        if img_name is not None:
            img_name_list = [img_name]
        else:
            if self.display_mode() == DisplayMode.PLANE_FIXED:
                img_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
            else:
                img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]

        for i in img_name_list:
            self.img_qlabel[i].refresh_crosses()

    def update_cross_id(
        self,
        cross: DraggableCross,
        old_cross_id: tuple,
        new_cross_id: tuple,
    ):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                cross = self.img_qlabel[i].get_cross_by_id(old_cross_id)
                cross.cross_id = new_cross_id
        else:
            cross.cross_id = new_cross_id

    def remove_3d_pos_of_selected_cross(self, cross: DraggableCross):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.gtvt_click_pos_3d = None

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            pos_3d = cross.cross_id
            if pos_3d in self.gtvn_clicks_pos_3d:
                self.gtvn_clicks_pos_3d.remove(pos_3d)

    def set_crosses_dragging_offset(self, img_qlabel: CustomQLabel, pos: QPoint):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_qlabel[i].selected_cross.offset = pos
        else:
            img_qlabel.selected_cross.offset = pos

    def set_crosses_dragging_state(self, img_qlabel: CustomQLabel, dragging: bool):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_qlabel[i].selected_cross.dragging = dragging
        else:
            img_qlabel.selected_cross.dragging = dragging

    def move_cross(self, img_qlabel: CustomQLabel, pos: QPoint):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_qlabel[i].selected_cross.move(pos)
        else:
            img_qlabel.selected_cross.move(pos)

    def delete_selected_crosses(self):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]

        for i in img_name_list:
            self.img_qlabel[i].delete_selected_cross()

    def select_cross(self, cross_id: tuple):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
        for i in img_name_list:
            self.img_qlabel[i].select_cross(cross_id)

    def get_nii_spacing(self):
        return self._nii_spacing

    def get_3d_img_shape(self):
        if self.img_3d[Modal.CT] is not None:
            return self.img_3d[Modal.CT].shape
        else:
            return None

    def __init__(
        self,
        idl_remark: str = None,
        debug_mode: bool = False,
    ):
        # pass debug_mode parameter to the parent class
        super().__init__(idl_remark=idl_remark, debug_mode=debug_mode)

    def _init_widgets_annotation(self):
        # text label
        for i in ["gtvt.progress", "gtvn.progress", "draw.size"]:
            self._text_label[i] = QtWidgets.QLabel()

        # set text
        self._text_label["draw.size"].setText("Pen Size")
        self._text_label["gtvt.progress"].setText("Generating GTVt")
        self._text_label["gtvn.progress"].setText("Generating GTVn")
        for i in ["gtvt", "gtvn"]:
            self._text_label["{}.progress".format(i)].hide()

        # radio button
        for i in ["gtvt", "gtvn"]:
            self._radio_btn["correct.{}".format(i)] = QtWidgets.QRadioButton()
        self._radio_btn["correct.gtvt"].setText("Correct GTVt")
        self._radio_btn["correct.gtvn"].setText("Correct GTVn")
        self._radio_btn["correct.gtvt"].setChecked(True)

        # annotation buttons
        self.__btn = Dict()
        for i in ["pen", "eraser", "clear", "confirm"]:
            self.__btn[i] = QtWidgets.QPushButton()
            self.__btn[i].setFixedWidth(50)
            self.__btn[i].setFixedHeight(40)
            # set btn icons
            icon = QtGui.QIcon(os.path.join(g.PROJ_DIR, "icons", "{}.png".format(i)))
            if i == "pen":
                self.__btn[i].setIconSize(QSize(24, 24))
            elif i == "eraser":
                self.__btn[i].setIconSize(QSize(31, 31))
            else:
                self.__btn[i].setIconSize(QSize(25, 25))
            self.__btn[i].setIcon(icon)

        # connect btns to functions
        self.__btn["pen"].clicked.connect(self.__on_btn_pen_clicked)
        self.__btn["eraser"].clicked.connect(self.__on_btn_eraser_clicked)
        self.__btn["clear"].clicked.connect(self.__on_btn_clear_clicked)
        self.__btn["confirm"].clicked.connect(self.__on_btn_confirm_clicked)

        # gtvt/gtvn progress bars
        self.__progress_bar = Dict()
        for i in ["gtvt", "gtvn"]:
            self.__progress_bar[i] = QtWidgets.QProgressBar()
            self.__progress_bar[i].setRange(0, 100)
            self.__progress_bar[i].setValue(0)
            self.__progress_bar[i].hide()

        # pen size slider
        self._slider["draw.size"] = QtWidgets.QSlider()
        self._slider["draw.size"].setOrientation(Qt.Horizontal)
        self._slider["draw.size"].setMinimum(1)
        self._slider["draw.size"].setMaximum(7)
        self._slider["draw.size"].setValue(4)

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
        # self._collap["annotation"].setFixedHeight(300)
        v_layout = QtWidgets.QVBoxLayout()

        # add buttons
        h_layout = QtWidgets.QHBoxLayout()
        for i in ["pen", "eraser", "clear", "confirm"]:
            h_layout.addWidget(self.__btn[i])
        v_layout.addLayout(h_layout)

        # add draw size slider
        v_layout.addWidget(self._text_label["draw.size"])
        v_layout.addWidget(self._slider["draw.size"])

        # add progress bars
        for i in ["gtvt", "gtvn"]:
            v_layout.addWidget(self._text_label["{}.progress".format(i)])
            v_layout.addWidget(self.__progress_bar[i])

        # add radio buttons
        h_layout = QtWidgets.QHBoxLayout()
        for i in ["gtvt", "gtvn"]:
            h_layout.addWidget(self._radio_btn["correct.{}".format(i)])
        v_layout.addLayout(h_layout)

        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._collap["annotation"].addWidget(container)

    def _init_widgets_set_fonts(self):
        super()._init_widgets_set_fonts()
        for i in ["gtvt", "gtvn"]:
            self._radio_btn["correct.{}".format(i)].setFont(self._font_bold)

    def _init_widgets(self):
        super()._init_widgets()

        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._collap[i].hide()

        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].setEnabled(False)
            self._collap[i].collapse()

    def clear_gtvt_click_pos_3d(self):
        self.gtvt_click_pos_3d = None

    def clear_gtvn_clicks_pos_3d(self):
        self.gtvn_clicks_pos_3d = List()

    def _clear_img_3d(self):
        super()._clear_img_3d()
        for i in ["gtvt.correction.mask", "gtvn.correction.mask"]:
            self.img_3d[i] = None

    def _init_data(self, idl_remark: str = None, debug_mode: bool = False):
        super()._init_data()
        self.__debug_mode = debug_mode

        # init baseline id and idl.gtvt/gtvn id, keep them unchanged
        # (1) baseline id
        self._baseline_id = "baseline_real.idl"
        # (2) idl.gtvt/gtvn id
        cur_time = Timer.cur_time_str()
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

        # create idl.gtvt/gtvn folders
        for i in ["gtvt", "gtvn"]:
            Dir.create(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i])
            )

        # init and save idl step of all patients
        self.__idl_step_of_all_patients = Dict()
        for patient in self._patients.to_list():
            self.__idl_step_of_all_patients[
                "patient={}".format(patient)
            ] = IDLStep.CLICK_GTVT_CENTER
        self.__save_idl_step_of_all_patients()

        # initialize the position of gtvt click / gtvn clicks
        self.clear_gtvt_click_pos_3d()
        self.clear_gtvn_clicks_pos_3d()

        # drawing
        self.drawing_mode = DrawingMode.GTVT_PEN
        self.paint_pos = None  # Store the last painted point
        self.__gtvt_annotated_status = Dict()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.__gtvt_annotated_status[plane] = False

        # idl gtvt/gtvn thread
        self.__idl_gtvt_thread = IDLGTVtThread()
        self.__idl_gtvt_thread.progress_signal.connect(
            self.__update_idl_gtvt_progress_bar
        )
        self.__idl_gtvt_thread.complete_signal.connect(
            self.__on_idl_gtvt_thread_finished
        )
        self.__idl_gtvn_thread = IDLGTVnThread()
        self.__idl_gtvn_thread.progress_signal.connect(
            self.__update_idl_gtvn_progress_bar
        )
        self.__idl_gtvn_thread.complete_signal.connect(
            self.__on_idl_gtvn_thread_finished
        )

    def __save_idl_step_of_all_patients(self):
        for i in ["gtvt", "gtvn"]:
            idl_step_json_path = os.path.join(
                g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i], "idl_step.json"
            )
            Json.save(self.__idl_step_of_all_patients, idl_step_json_path)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_F12:
            pass

        # delete selected cross
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected_crosses()

        super().keyPressEvent(event)

    def __clear_all_drawing_layers(self, img_qlabel: CustomQLabel):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [img_qlabel.plane]
        for i in img_name_list:
            self.img_qlabel[i].drawing_layer = QtGui.QPixmap(self.img_qlabel[i].size())
            self.img_qlabel[i].drawing_layer.fill(Qt.transparent)
            self.img_qlabel[i].update()

    # this function is connected to widget, dont set input params to this function
    def __switch_drawing_mode_gtv(self):
        if self._radio_btn["correct.gtvt"].isChecked():
            if self.drawing_mode == DrawingMode.GTVN_PEN:
                self.drawing_mode = DrawingMode.GTVT_PEN
            elif self.drawing_mode == DrawingMode.GTVN_ERASER:
                self.drawing_mode = DrawingMode.GTVT_ERASER
        elif self._radio_btn["correct.gtvn"].isChecked():
            if self.drawing_mode == DrawingMode.GTVT_PEN:
                self.drawing_mode = DrawingMode.GTVN_PEN
            elif self.drawing_mode == DrawingMode.GTVT_ERASER:
                self.drawing_mode = DrawingMode.GTVN_ERASER

    def get_pen_size(self):
        return self._slider["draw.size"].value()

    def _load_baseline_data(self):
        # self._reset_zoomin()
        self._clear_img_3d()
        self._clear_img_qlabels()

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self._combox["patient"].setCurrentIndex(-1)  # show nothing

    def _add_msg_on_qimg(self, qimg: QtGui.QImage):
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

        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            text = "Please click the center of primary Gross Tumor Volumes (GTVt)"
        elif idl_step == IDLStep.DRAW_GTVT:
            text = "Please delineate GTVt in 3 anatomical planes"
        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            text = "Please click the center of malignant lymph nodes (GTVn)"
        elif idl_step == IDLStep.WAITING:
            text = "Neural Network is generating auto-segmentation, please wait..."
        elif idl_step == IDLStep.CORRECT_GTVT:
            text = "Please correct the GTVt auto-segmentation"
        elif idl_step == IDLStep.CORRECT_GTVN:
            text = "Please correct the GTVn auto-segmentation"
        elif idl_step == IDLStep.CORRECT_BOTH:
            text = "Please correct the GTVt and GTVn auto-segmentations"

        self._qimg_draw_text(
            qimg=qimg,
            text=text,
            pos=(pos_x, pos_y),
            color=self._color["green"],
        )

        # add info of annotated status for drawing gtvt
        if idl_step == IDLStep.DRAW_GTVT:
            pos_y += 5
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
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

    # rewrite this function (do nothing)
    def _add_score_on_qimg(self, qimg: QtGui.QImage):
        pass

    def _add_contour_description_on_qimg(
        self,
        qimg: QtGui.QImage,
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

    def _load_patient_data(self):
        # enable collapse bar
        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].setEnabled(True)
        # expand collapse bar
        for i in ["annotation", "display.mode"]:
            self._collap[i].expand()

        # disable pen and eraser as clicking gtvt center dont need them
        for i in ["pen", "eraser"]:
            self.__btn[i].setEnabled(False)
        # enable buttons
        for i in ["clear", "confirm"]:
            self.__btn[i].setEnabled(True)

        # restore default cursor
        self.setCursor(Qt.ArrowCursor)

        # hide pen/eraser size
        self._slider["draw.size"].hide()
        self._text_label["draw.size"].hide()

        # clear data
        self._clear_img_3d()
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
        self.reset_cur_slice_id()
        self._load_idl_gtvt_data()
        self._load_idl_gtvn_data()
        self.refresh_img_qlabels()
        self.refresh_crosses_on_qlabels()

    def reset_cur_slice_id(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER or idl_step == IDLStep.DRAW_GTVT:
            if self.gtvt_click_pos_3d is None:
                self.cur_slice_id[Plane.TRANSVERSE] = (
                    self.img_3d[Modal.CT].shape[0] // 2
                )
                self.cur_slice_id[Plane.CORONAL] = self.img_3d[Modal.CT].shape[1] // 2
                self.cur_slice_id[Plane.SAGITTAL] = self.img_3d[Modal.CT].shape[2] // 2
            else:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            if len(self.gtvn_clicks_pos_3d) == 0:
                self.cur_slice_id = self.__get_gtvt_center_slices_id()
            else:
                self.cur_slice_id = self.__get_gtvn_center_slices_id()

        elif idl_step in [
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

    def get_cur_patient_idl_step(self):
        return self.__idl_step_of_all_patients["patient={}".format(self._cur_patient)]

    def update_cur_patient_idl_step(self, step: str):
        self.__idl_step_of_all_patients["patient={}".format(self._cur_patient)] = step
        self.__save_idl_step_of_all_patients()

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
