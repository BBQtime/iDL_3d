import os

import cv2
import numpy as np
import qimage2ndarray
from custom import GPU, Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Timer, Value
from numpy import ndarray
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QEvent, QPoint, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMessageBox
from scipy import ndimage
from str_lib import DisplayMode, DrawingMode, IDLStep, Modal, Plane
from superqt import QCollapsible
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt
from ui_draggable_cross import DraggableCross
from ui_idl_step_label import IDLStepLabel
from ui_img_box import ImgBox
from ui_replay import UiReplay


class IDLThread(QThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

    def __init__(
        self,
        progress_bar: QtWidgets.QProgressBar,
        progress_bar_label: QtWidgets.QLabel,
    ):
        super().__init__()
        self.is_running = False
        self._progress_bar = progress_bar
        self._progress_bar_label = progress_bar_label

    def stop(self):
        if self.is_running:
            self.terminate()  # force stop
            GPU.clear_cache()
            # self.quit()  # signal the thread to exit its event loop
            # self.wait()  # wait for the thread to be cleaned up properly
            self.is_running = False
            self._hide_progress_widgets()

    def _show_progress_widgets(self):
        self._progress_bar_label.show()
        self._progress_bar.setValue(0)
        self._progress_bar.show()

    def _hide_progress_widgets(self):
        self._progress_bar_label.hide()
        self._progress_bar.hide()
        self._progress_bar.setValue(0)


class IDLGTVnThread(IDLThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

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

    def run(self):
        self._show_progress_widgets()
        self.is_running = True
        training_idl_gtvn = TrainingIDLGTVn(self.progress_signal)
        training_idl_gtvn.real_idl(
            idl_gtvn_id=self.__idl_gtvn_id,
            patient=self.__patient,
            idl_gtvn_clicks=self.__idl_gtvn_clicks,
            dataset_part=self.__dataset_part,
            dataset_ver=self.__dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self._hide_progress_widgets()
        self.is_running = False
        self.complete_signal.emit()


class IDLGTVtThread(IDLThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

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

    def run(self):
        self._show_progress_widgets()
        self.is_running = True
        training_idl_gtvt = TrainingIDLGTVt(self.progress_signal)
        training_idl_gtvt.real_idl(
            idl_gtvt_id=self.__idl_gtvt_id,
            patient=self.__patient,
            dataset_ver=self.__dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self._hide_progress_widgets()
        self.is_running = False
        self.complete_signal.emit()


class UiIDL(UiReplay):
    def draw_on_img_boxes_press(self, event: QtGui.QMouseEvent):
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
                self.refresh_imgs()

        elif idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            self.paint_pos = event.pos()

    def draw_on_img_boxes_move(self, event: QtGui.QMouseEvent, img_box: ImgBox):
        if self.paint_pos is None:
            return

        pen_size = self.get_pen_size()
        eraser_size = pen_size + 2
        eraser_color = QtGui.QColor(*self._color["eraser"])

        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [img_box.plane]
        for i in img_name_list:
            painter = QtGui.QPainter(self.img_box[i].drawing_layer)

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
                self.img_box[i].pen_mode = False
            elif self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVN_PEN]:
                painter.setPen(
                    QtGui.QPen(pen_color, pen_size, Qt.SolidLine, Qt.RoundCap)
                )
                self.img_box[i].pen_mode = True

            painter.drawLine(self.paint_pos, event.pos())

            self.img_box[i].update()  # schedule a repaint

        self.paint_pos = event.pos()  # update paint pos

    def draw_on_img_boxes_release(self, img_box: ImgBox):
        if self.paint_pos is None:
            return

        # binarize threshold
        # this is for saving qimage as ndarray
        # binarization is needed before and after resize the ndarray
        binary_threshold = 0.5

        # save drawing layer into 2d ndarray
        # qpixmap to a qimage
        qimg = img_box.drawing_layer.toImage()
        # qimage to ndarray
        annotation_2d = qimage2ndarray.alpha_view(qimg).astype(np.float32)
        annotation_2d /= 255

        # binarization (before resize)
        annotation_2d = Img.binarize(img=annotation_2d, threshold=binary_threshold)

        # crop annotation_2d based on roi
        x = img_box.roi.x
        y = img_box.roi.y
        width = img_box.roi.width
        height = img_box.roi.height
        annotation_2d = annotation_2d[y : y + height, x : x + width]

        # resize to actual size
        if img_box.plane == Plane.SAGITTAL:
            actual_shape = self.img_3d[Modal.CT][:, :, 0].shape
        elif img_box.plane == Plane.CORONAL:
            actual_shape = self.img_3d[Modal.CT][:, 0, :].shape
        elif img_box.plane == Plane.TRANSVERSE:
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
            if img_box.plane == Plane.TRANSVERSE:
                segment = self.img_3d["gtvt.annotation"][t, :, :]
            elif img_box.plane == Plane.CORONAL:
                segment = self.img_3d["gtvt.annotation"][:, c, :]
            elif img_box.plane == Plane.SAGITTAL:
                segment = self.img_3d["gtvt.annotation"][:, :, s]
        # (2)correction
        elif idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            t = c = s = self.cur_slice_id[img_box.plane]
            if self.drawing_mode in [DrawingMode.GTVT_PEN, DrawingMode.GTVT_ERASER]:
                gtv = "gtvt"
            elif self.drawing_mode in [DrawingMode.GTVN_PEN, DrawingMode.GTVN_ERASER]:
                gtv = "gtvn"
            _3d_img = self.img_3d["{}.pred.final".format(gtv)]
            if img_box.plane == Plane.TRANSVERSE:
                segment = _3d_img[t, :, :].copy()
            elif img_box.plane == Plane.CORONAL:
                segment = _3d_img[:, c, :].copy()
            elif img_box.plane == Plane.SAGITTAL:
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
        if img_box.plane == Plane.TRANSVERSE:
            _3d_img[t, :, :] = segment
        elif img_box.plane == Plane.CORONAL:
            _3d_img[:, c, :] = segment
        elif img_box.plane == Plane.SAGITTAL:
            _3d_img[:, :, s] = segment

        # update masks then save corrections and masks
        if idl_step in [
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
            IDLStep.CORRECT_BOTH,
        ]:
            # (1)update correction masks
            if img_box.plane == Plane.TRANSVERSE:
                if segment.max() == 0:
                    _3d_mask[t, :, :] = np.zeros_like(segment)
                else:
                    _3d_mask[t, :, :] = np.ones_like(segment)
            elif img_box.plane == Plane.CORONAL:
                if segment.max() == 0:
                    _3d_mask[:, c, :] = np.zeros_like(segment)
                else:
                    _3d_mask[:, c, :] = np.ones_like(segment)
            elif img_box.plane == Plane.SAGITTAL:
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
        self.__clear_all_drawing_layers(img_box)
        self.refresh_imgs()

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

        # for i in [
        #     IDLStep.DRAW_GTVT_TRANSVERSE,
        #     IDLStep.DRAW_GTVT_CORONAL,
        #     IDLStep.DRAW_GTVT_SAGITTAL,
        # ]:
        #     self._text_label[i].hide()

        # button size 562*187
        btn_h = 27
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
        v_layout.setSpacing(3)
        v_layout.addWidget(self._btn["next.step"], alignment=Qt.AlignmentFlag.AlignRight)

        for i in self._text_label.keys():
            if i in idl_step_list:
                v_layout.addWidget(self._text_label[i])

        # container
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        container.setFixedHeight(290)
        self._add_border(container)

        # create qcollapsible space
        self._collap["todo.list"] = QCollapsible("TODO LIST")
        self._collap["todo.list"].addWidget(container)
        self._collap["todo.list"].expand()

    def __goto_idl_step_click_gtvt_center(self):
        # stop idl qthreads (if running)
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # disable pen and eraser as clicking gtvt center dont need them
        for i in ["pen", "eraser"]:
            self._btn[i].setEnabled(False)

        # enable buttons
        for i in ["clear", "next.step"]:
            self._btn[i].setEnabled(True)

        # restore default cursor
        self.setCursor(Qt.ArrowCursor)

        # hide pen/eraser size
        self._slider["draw.size"].hide()
        self._text_label["draw.size"].hide()

        # update todolist
        self._text_label[IDLStep.SELECT_PATIENT].set_status_done()
        self._text_label[IDLStep.CLICK_GTVT_CENTER].set_status_ongoing()
        for i in [
            IDLStep.DRAW_GTVT,
            IDLStep.DRAW_GTVT_TRANSVERSE,
            IDLStep.DRAW_GTVT_CORONAL,
            IDLStep.DRAW_GTVT_SAGITTAL,
            IDLStep.CLICK_GTVN_CENTER,
            IDLStep.WAITING,
            IDLStep.CORRECT_GTVT,
            IDLStep.CORRECT_GTVN,
        ]:
            self._text_label[i].set_status_notstart()

        # clear images
        img_name_list = ["gtvt.click", "gtvn.clicks", "gtvt.annotation"]
        for i in ["gtvt", "gtvn"]:
            img_name_list += [
                "{}.pred".format(i),
                "{}.correction".format(i),
                "{}.correction.mask".format(i),
                "{}.pred.final".format(i),
            ]
        for i in img_name_list:
            self.img_3d[i] = None

        # clear gtvt click pos
        self.gtvt_click_pos_3d = None

        # DO NOT clear self.gtvn_clicks_pos_3d

        # refresh images and crosses
        self.refresh_imgs()
        self.refresh_crosses()

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

        # (5) goto next step
        self.__goto_idl_step_draw_gtvt()

    def __goto_idl_step_draw_gtvt(self):
        # stop idl qthread (if running)
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # update widgets
        self.__set_mouse_cursor("pen")

        # enable pen and eraser for gtvt delineation
        for i in ["pen", "eraser"]:
            self._btn[i].setEnabled(True)
        self._slider["draw.size"].show()
        self._text_label["draw.size"].show()

        # clear images
        self.img_3d["gtvt.annotation"] = np.zeros_like(self.img_3d[Modal.CT])
        img_name_list = ["gtvn.clicks"]
        for i in ["gtvt", "gtvn"]:
            img_name_list += [
                "{}.pred".format(i),
                "{}.correction".format(i),
                "{}.correction.mask".format(i),
                "{}.pred.final".format(i),
            ]
        for i in img_name_list:
            self.img_3d[i] = None

        # DO NOT clear self.gtvt_click_pos_3d and self.gtvn_clicks_pos_3d

        # update status
        self.update_cur_patient_idl_step(IDLStep.DRAW_GTVT)
        self.drawing_mode = DrawingMode.GTVT_PEN

        # remove crosses and refresh images
        self.delete_all_crosses()
        self.refresh_imgs()

    def __confirm_gtvt_annotation(self):
        # (1) check gtvt annotated statues
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            if self.__gtvt_annotated_status[plane] is False:
                QMessageBox.information(
                    self,
                    "Information",
                    "Please draw GTVt in {} plane.".format(plane),
                    QMessageBox.Ok,
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

        # (3)start idl gtvt thread
        self.__idl_gtvt_thread.set_param(
            idl_gtvt_id=self._idl_id["gtvt"],
            patient=self._cur_patient,
            dataset_ver=self._dataset_ver,
            debug_mode=self.__debug_mode,
        )
        self.__idl_gtvt_thread.start()

        # (4) goto next step
        self.__goto_idl_step_click_gtvn_center()

    def __goto_idl_step_click_gtvn_center(self):
        # stop idl gtvn qthread (if running)
        self.__idl_gtvn_thread.stop()

        # restore default cursor
        self.setCursor(Qt.ArrowCursor)
        for i in ["pen", "eraser"]:
            self._btn[i].setEnabled(False)
        self._slider["draw.size"].hide()
        self._text_label["draw.size"].hide()

        # clear images
        for i in [
            "gtvn.clicks",
            "gtvn.pred",
            "gtvn.correction",
            "gtvn.correction.mask",
            "gtvn.pred.final",
        ]:
            self.img_3d[i] = None

        # clear gtvn clicks
        self.gtvn_clicks_pos_3d = List()

        # update status (before refresh images)
        self.update_cur_patient_idl_step(IDLStep.CLICK_GTVN_CENTER)

        # refresh images and crosses
        self.refresh_imgs()
        self.refresh_crosses()

    def on_idl_step_text_box_clicked(self, text_box: IDLStepLabel):
        cur_idl_step = self.get_cur_patient_idl_step()

        # jump to SELECT_PATIENT
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

        # jump to CLICK_GTVT_CENTER
        elif text_box == self._text_label[IDLStep.CLICK_GTVT_CENTER]:
            if cur_idl_step not in [
                IDLStep.DRAW_GTVT,
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
                IDLStep.CORRECT_BOTH,
            ]:
                return

            text = (
                "Would you like to jump to SETP 2 and re-click the center of GTVt? "
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

            self.update_cur_patient_idl_step(IDLStep.CLICK_GTVT_CENTER)
            self.__goto_idl_step_click_gtvt_center()

        # jump to DRAW_GTVT
        elif text_box == self._text_label[IDLStep.DRAW_GTVT]:
            if cur_idl_step not in [
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
                IDLStep.CORRECT_BOTH,
            ]:
                return

            text = (
                "Would you like to jump to SETP 3 and re-delineate GTVt? "
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

            self.update_cur_patient_idl_step(IDLStep.DRAW_GTVT)
            self.__goto_idl_step_draw_gtvt()

        # jump to CLICK_GTVN_CENTER
        elif text_box == self._text_label[IDLStep.CLICK_GTVN_CENTER]:
            if cur_idl_step not in [
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
                IDLStep.CORRECT_BOTH,
            ]:
                return

            text = (
                "Would you like to jump to SETP 4 and re-click the centers of GTVn? "
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

            self.update_cur_patient_idl_step(IDLStep.CLICK_GTVN_CENTER)
            self.__goto_idl_step_click_gtvn_center()

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

    def __update_idl_gtvt_progress_bar(self, progress_signal: float):
        progress_int = round(progress_signal * 100)
        Value.limit_range(progress_int, (0, 100))
        self.__progress_bar["gtvt"].setValue(progress_int)

    def __on_idl_gtvt_thread_finished(self):
        # update idl step and widgets first

        # (1) idl.gtvn thread is not running
        if not self.__idl_gtvn_thread.is_running:
            # do nothing if user has not submitted gtvn clicks
            if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
                pass
            # gtvn thread is over, correct both gtvt/gtvn
            else:
                self.update_cur_patient_idl_step(IDLStep.CORRECT_BOTH)
                # show radio buttons
                for i in ["gtvt", "gtvn"]:
                    self._radio_btn["correct.{}".format(i)].show()
                # dont change drawing mode and radio buttons, because user is correcting gtvn

        # (2) idl.gtvn thread is running, only correct gtvt
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
                    self._btn[i].setEnabled(True)
                self._slider["draw.size"].show()
                self._text_label["draw.size"].show()

            # idl step can not be other values
            else:
                Debug.error_exit("idl step error")

        # update gtvt 3d imgs and img_boxes
        self._load_idl_gtvt_data()
        self.__combine_pred_annotation_correction()
        self.refresh_imgs()
        # init correction and mask
        for i in ["gtvt.correction", "gtvt.correction.mask"]:
            self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])

    def __confirm_gtvn_center(self):
        if len(self.gtvn_clicks_pos_3d) == 0:
            reply = QMessageBox.question(
                self,
                "Message",
                """No GTVn clicks detected.
                If current patient does not have GTVn, choose "Yes" to continue.
                Choose "No" to return and click GTVn.""",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        # (1)add clicks into 3d img
        if self.img_3d["gtvn.clicks"] is None:
            self.img_3d["gtvn.clicks"] = np.zeros_like(self.img_3d[Modal.CT])
        for pos in self.gtvn_clicks_pos_3d:
            # pos 0-transverse 1-coronal 2-saggital
            self.img_3d["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1

        # (2) update widget before idl.gtvn thread (for better user experience)
        # gtvn thread is still running, only correct gtvt
        if not self.__idl_gtvt_thread.is_running:
            # update idl step before refresh img_boxes
            self.update_cur_patient_idl_step(IDLStep.CORRECT_GTVT)
            self._radio_btn["correct.gtvt"].setChecked(True)
            self.__set_mouse_cursor("pen")
            self.drawing_mode = DrawingMode.GTVT_PEN
            for i in ["pen", "eraser", "clear"]:
                self._btn[i].setEnabled(True)
            self._slider["draw.size"].show()
            self._text_label["draw.size"].show()

        # gtvt thread is not completedgoto waiting step
        else:
            # update idl step before refresh img_boxes
            self.update_cur_patient_idl_step(IDLStep.WAITING)
            self._btn["clear"].setEnabled(False)

        # disable "next.step" button, its not needed anymore
        self._btn["next.step"].setEnabled(False)

        #  delete cross and refresh to show gtvn clicks
        self.delete_all_crosses()
        self.refresh_imgs()

        # (3) transform gtvn clicks for idl.gtvn thread
        # copy data (dont change origin ndarray)
        idl_gtvn_clicks = self.img_3d["gtvn.clicks"].copy()
        # no need to flip empty img
        if idl_gtvn_clicks.max() > 0:
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
        if not self.__idl_gtvt_thread.is_running:
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
                self._btn[i].setEnabled(True)
            self._slider["draw.size"].show()
            self._text_label["draw.size"].show()

        # show gtvt/gtvn switch radio buttons
        if self.get_cur_patient_idl_step() == IDLStep.CORRECT_BOTH:
            for i in ["gtvt", "gtvn"]:
                self._radio_btn["correct.{}".format(i)].show()

        # update 3d imgs and img_boxes
        self._load_idl_gtvn_data()
        self.__combine_pred_annotation_correction()
        self.refresh_imgs()
        # init correction and mask
        for i in ["gtvn.correction", "gtvn.correction.mask"]:
            self.img_3d[i] = np.zeros_like(self.img_3d[Modal.CT])

    # this function is connected to widget, dont set input params to this function
    def __on_btn_next_step_clicked(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.__confirm_gtvt_center()
        elif idl_step == IDLStep.DRAW_GTVT:
            self.__confirm_gtvt_annotation()
        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.__confirm_gtvn_center()

    # check annotation in 3 different planes
    def __update_gtvt_annotated_status(self) -> Dict:
        if self.img_3d["gtvt.annotation"] is None:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                self.__gtvt_annotated_status[plane] = False
                self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()

        else:
            t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
            t, c, s = int(t), int(c), int(s)
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                if plane == Plane.TRANSVERSE:
                    cur_plane_annotation = self.img_3d["gtvt.annotation"][
                        t, :, :
                    ].copy()
                    cur_plane_annotation[c, :] = 0
                    cur_plane_annotation[:, s] = 0

                elif plane == Plane.CORONAL:
                    cur_plane_annotation = self.img_3d["gtvt.annotation"][
                        :, c, :
                    ].copy()
                    cur_plane_annotation[t, :] = 0
                    cur_plane_annotation[:, s] = 0

                elif plane == Plane.SAGITTAL:
                    cur_plane_annotation = self.img_3d["gtvt.annotation"][
                        :, :, s
                    ].copy()
                    cur_plane_annotation[t, :] = 0
                    cur_plane_annotation[:, c] = 0

                if cur_plane_annotation.max() == 0:
                    self.__gtvt_annotated_status[plane] = False
                    self._text_label["draw.gtvt.{}".format(plane)].set_status_missing()
                else:
                    self.__gtvt_annotated_status[plane] = True
                    self._text_label["draw.gtvt.{}".format(plane)].set_status_done()

    def refresh_imgs(self, img_name=None):
        # no patient loaded
        if self.img_3d[Modal.CT] is None:
            if self.display_mode() == DisplayMode.PLANE_FIXED:
                img_name = Plane.TRANSVERSE
            else:
                img_name = Modal.CT
            # ask user to select a patient
            w = self.img_box[img_name].width()
            h = self.img_box[img_name].height()
            qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB888)
            black = QtGui.QColor(0, 0, 0)
            qimg.fill(black)
            self._add_msg_on_qimg(qimg)
            self.img_box[img_name].set_background(qimg)
            self.img_box[img_name].update()
            return

        super().refresh_imgs(replay_mode=False, img_name=img_name)

    def __change_color(self, pixmap: QtGui.QPixmap, old_color, new_color):
        image = pixmap.toImage()
        old_qcolor = QtGui.QColor(*old_color)  # Unpack the tuple
        new_qcolor = QtGui.QColor(*new_color)  # Unpack the tuple

        for x in range(image.width()):
            for y in range(image.height()):
                if image.pixelColor(x, y) == old_qcolor:
                    image.setPixelColor(x, y, new_qcolor)
        return QtGui.QPixmap.fromImage(image)

    def __set_mouse_cursor(self, cursor_type: str):
        if cursor_type not in ["pen", "eraser"]:
            Debug.error_exit("'cursor_type' must be one of 'pen' or 'eraser'!")

        cursor_size = 32  # no larger than 32
        cursor_pixmap = QtGui.QPixmap(
            (os.path.join(g.PROJ_DIR, "icons", "{}_cursor.png".format(cursor_type)))
        )

        # # change color
        # old_color = (2, 252, 240)
        # cursor_pixmap = self.__change_color(
        #     pixmap=cursor_pixmap,
        #     old_color=old_color,
        #     new_color=self._color["gtvt.pred"],
        # )

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
        self._color["eraser"] = self._color["black"]  # transparent
        self._color["gtvt.pred.final"] = self._color["gtvt.pred"]
        self._color["gtvn.pred.final"] = self._color["gtvn.pred"]

    # this function is connected to widget, dont set input params to this function
    def __on_btn_clear_clicked(self):
        idl_step = self.get_cur_patient_idl_step()

        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.gtvt_click_pos_3d = None
            self.refresh_crosses()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.gtvn_clicks_pos_3d = List()
            self.refresh_crosses()

        elif idl_step == IDLStep.DRAW_GTVT:
            # modality fixed mode: clear annotation on cur plane
            if self.display_mode() == DisplayMode.MODAL_FIXED:
                t, c, s = np.where(self.img_3d["gtvt.click"] == 1)
                t, c, s = int(t), int(c), int(s)
                # use mask to filter out the annotation on current anatomical plane
                if self.img_box[Modal.CT].plane == Plane.TRANSVERSE:
                    mask = np.zeros_like(self.img_3d["gtvt.annotation"][t, :, :])
                    mask[c, :] = 1
                    mask[:, s] = 1
                    self.img_3d["gtvt.annotation"][t, :, :] *= mask
                elif self.img_box[Modal.CT].plane == Plane.CORONAL:
                    mask = np.zeros_like(self.img_3d["gtvt.annotation"][:, c, :])
                    mask[t, :] = 1
                    mask[:, s] = 1
                    self.img_3d["gtvt.annotation"][:, c, :] *= mask
                elif self.img_box[Modal.CT].plane == Plane.SAGITTAL:
                    mask = np.zeros_like(self.img_3d["gtvt.annotation"][:, :, s])
                    mask[t, :] = 1
                    mask[:, c] = 1
                    self.img_3d["gtvt.annotation"][:, :, s] *= mask
                # update gtvt annotated status
                self.__gtvt_annotated_status[self.img_box[Modal.CT].plane] = False
                # update todo list
                self._text_label[
                    "draw.gtvt.{}".format(self.img_box[Modal.CT].plane)
                ].set_status_missing()

            # plane fixed mode: clear whole annotation
            else:
                self.img_3d["gtvt.annotation"] = np.zeros_like(self.img_3d[Modal.CT])
                # update gtvt annotated status
                for i in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                    self.__gtvt_annotated_status[i] = False
                # update todo list
                for i in [
                    IDLStep.DRAW_GTVT_TRANSVERSE,
                    IDLStep.DRAW_GTVT_CORONAL,
                    IDLStep.DRAW_GTVT_SAGITTAL,
                ]:
                    self._text_label[i].set_status_missing()

            self.__combine_pred_annotation_correction()
            self.refresh_imgs()

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
                t = c = s = self.cur_slice_id[self.img_box[Modal.CT].plane]
                _3d_img = self.img_3d["{}.correction".format(gtv)]
                _3d_mask = self.img_3d["{}.correction.mask".format(gtv)]
                if self.img_box[Modal.CT].plane == Plane.TRANSVERSE:
                    _3d_img[t, :, :] = np.zeros_like(_3d_img[t, :, :])
                    _3d_mask[t, :, :] = np.zeros_like(_3d_mask[t, :, :])
                elif self.img_box[Modal.CT].plane == Plane.CORONAL:
                    _3d_img[:, c, :] = np.zeros_like(_3d_img[:, c, :])
                    _3d_mask[:, c, :] = np.zeros_like(_3d_mask[:, c, :])
                elif self.img_box[Modal.CT].plane == Plane.SAGITTAL:
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
            self.refresh_imgs()

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_crosses()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.refresh_crosses()

    def _modal_fixed_mode_switch_plane(self, new_plane: str = None):
        super()._modal_fixed_mode_switch_plane(new_plane)
        self.refresh_crosses()

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
            self.img_box[i].delete_all_crosses()

    def refresh_crosses(self, img_name: str = None):
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
            self.img_box[i].refresh_crosses()

    def update_cross_id(
        self,
        cross: DraggableCross,
        old_cross_id: tuple,
        new_cross_id: tuple,
    ):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                cross = self.img_box[i].get_cross_by_id(old_cross_id)
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

    def set_crosses_dragging_offset(self, img_box: ImgBox, pos: QPoint):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_box[i].selected_cross.offset = pos
        else:
            img_box.selected_cross.offset = pos

    def set_crosses_dragging_state(self, img_box: ImgBox, dragging: bool):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_box[i].selected_cross.dragging = dragging
        else:
            img_box.selected_cross.dragging = dragging

    def move_cross(self, img_box: ImgBox, pos: QPoint):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                self.img_box[i].selected_cross.move(pos)
        else:
            img_box.selected_cross.move(pos)

    def delete_selected_crosses(self):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]

        for i in img_name_list:
            self.img_box[i].delete_selected_cross()

    def select_cross(self, cross_id: tuple):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
        for i in img_name_list:
            self.img_box[i].select_cross(cross_id)

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
        for i in ["pen", "eraser", "clear"]:
            self._btn[i] = QtWidgets.QPushButton()
            self._btn[i].setFixedWidth(50)
            self._btn[i].setFixedHeight(40)
            # set btn icons
            icon = QtGui.QIcon(os.path.join(g.PROJ_DIR, "icons", "{}.png".format(i)))
            if i == "pen":
                self._btn[i].setIconSize(QSize(24, 24))
            elif i == "eraser":
                self._btn[i].setIconSize(QSize(31, 31))
            elif i == "clear":
                self._btn[i].setIconSize(QSize(25, 25))
            self._btn[i].setIcon(icon)

        # connect btns to functions
        self._btn["pen"].clicked.connect(self.__on_btn_pen_clicked)
        self._btn["eraser"].clicked.connect(self.__on_btn_eraser_clicked)
        self._btn["clear"].clicked.connect(self.__on_btn_clear_clicked)

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
        v_layout = QtWidgets.QVBoxLayout()

        # add buttons
        h_layout = QtWidgets.QHBoxLayout()
        for i in ["pen", "eraser", "clear"]:
            h_layout.addWidget(self._btn[i])
        v_layout.addLayout(h_layout)

        # add draw size slider
        v_layout.addWidget(self._text_label["draw.size"])
        v_layout.addWidget(self._slider["draw.size"])

        # add progress bars and progress bar labels
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

        # add radio buttons
        h_layout = QtWidgets.QHBoxLayout()
        for i in ["gtvt", "gtvn"]:
            h_layout.addWidget(self._radio_btn["correct.{}".format(i)])
        v_layout.addLayout(h_layout)

        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self._add_border(container)
        self._collap["annotation"].addWidget(container)

    def _init_widgets_set_fonts(self):
        super()._init_widgets_set_fonts()
        for i in ["gtvt", "gtvn"]:
            self._radio_btn["correct.{}".format(i)].setFont(self._font)

    def _init_widgets(self):
        super()._init_widgets()

        for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
            self._collap[i].hide()

        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            self._collap[i].setEnabled(False)
            self._collap[i].collapse()

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
        self.gtvt_click_pos_3d = None
        self.gtvn_clicks_pos_3d = List()

        # drawing
        self.drawing_mode = DrawingMode.GTVT_PEN
        self.paint_pos = None  # Store the last painted point
        self.__gtvt_annotated_status = Dict()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            self.__gtvt_annotated_status[plane] = False

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

    def __clear_all_drawing_layers(self, img_box: ImgBox):
        if self.display_mode() == DisplayMode.MODAL_FIXED:
            img_name_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        else:
            img_name_list = [img_box.plane]
        for i in img_name_list:
            self.img_box[i].drawing_layer = QtGui.QPixmap(self.img_box[i].size())
            self.img_box[i].drawing_layer.fill(Qt.transparent)
            self.img_box[i].update()

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
        self._clear_img_boxes()

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self.combox["patient"].setCurrentIndex(-1)  # show nothing

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
        # stop idl qthreads (if running)
        self.__idl_gtvt_thread.stop()
        self.__idl_gtvn_thread.stop()

        # expand and enable collapse bar
        for i in ["annotation", "display.mode", "color.enhance", "zoom"]:
            if not self._collap[i].isEnabled():
                self._collap[i].setEnabled(True)
                if i in ["annotation", "display.mode"]:
                    self._collap[i].expand()

        # clear data
        self._clear_img_3d()
        self.gtvt_click_pos_3d = None
        self.gtvn_clicks_pos_3d = List()

        self._cur_patient = self.combox["patient"].currentText()
        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_dir_and_nii_spacing()

        # self._reset_zoomin()

        # load multi-modal imgs only, no labels
        self._load_multi_modal_imgs()

        # load idl gtvt/gtvn images
        self._load_idl_gtvt_data()
        self._load_idl_gtvn_data()

        # (1) call reset_cur_slice_id() after _load_multi_modal_imgs
        # (2) call reset_cur_slice_id() after after _load_idl_gtvt_data() and _load_idl_gtvn_data()
        # because reset_cur_slice_id() will need gtvt_click_pos_3d and gtvn_clicks_pos_3d
        # and these are loaded by _load_idl_gtvt_data() and _load_idl_gtvn_data()
        self.reset_cur_slice_id()

        # jump to specific idl step
        idl_step = self.get_cur_patient_idl_step()
        if idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.__goto_idl_step_click_gtvt_center()

        elif idl_step == IDLStep.DRAW_GTVT:
            self.__goto_idl_step_draw_gtvt()

        elif idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.__goto_idl_step_click_gtvn_center()

        elif idl_step == IDLStep.CORRECT_BOTH:
            self.__goto_idl_step_correct_both()

        elif idl_step in [IDLStep.WAITING, IDLStep.CORRECT_GTVT, IDLStep.CORRECT_GTVN]:
            Debug.error_exit("_load_patient_data(): IDL Step error!")

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
        if self._cur_patient is None:
            return None
        else:
            return self.__idl_step_of_all_patients[
                "patient={}".format(self._cur_patient)
            ]

    def update_cur_patient_idl_step(self, cur_step: str):
        # update status
        self.__idl_step_of_all_patients[
            "patient={}".format(self._cur_patient)
        ] = cur_step
        # save json
        self.__save_idl_step_of_all_patients()

        # update todo list
        if cur_step != IDLStep.CORRECT_BOTH:
            self._text_label[cur_step].set_status_ongoing()

        if cur_step == IDLStep.CLICK_GTVT_CENTER:
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
        elif cur_step == IDLStep.DRAW_GTVT:
            done_step_list = [IDLStep.SELECT_PATIENT, IDLStep.CLICK_GTVT_CENTER]
            notstart_step_list = [
                IDLStep.CLICK_GTVN_CENTER,
                IDLStep.WAITING,
                IDLStep.CORRECT_GTVT,
                IDLStep.CORRECT_GTVN,
            ]
            # sub steps of draw.gtvt
            self.__update_gtvt_annotated_status()

        elif cur_step == IDLStep.CLICK_GTVN_CENTER:
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
            self._text_label[IDLStep.WAITING].set_status_ongoing()

        elif cur_step == IDLStep.WAITING:
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
        elif cur_step == IDLStep.CORRECT_GTVT:
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

        elif cur_step == IDLStep.CORRECT_GTVN:
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

        elif cur_step == IDLStep.CORRECT_BOTH:
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
