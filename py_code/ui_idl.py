import os
import random

import numpy as np
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import IDLStep, Img, Json, List, Nii, Plane, Time
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QIcon, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from training_idl_gtvn import TrainingIDLGTVn
from ui_draggable_cross import DraggableCross
from ui_replay import UiReplay

# always fill gtvt


class UiIDL(UiReplay):
    def draw_on_4_qlabels_press(self, event: QMouseEvent):
        self.paint_pos = event.pos()

    def draw_on_4_qlabels_move(self, event: QMouseEvent):
        if self.paint_pos is None:
            return
        pen_size = self.get_pen_size()
        for i in ["ct", "pt", "mr1", "mr2"]:
            painter = QPainter(self.img_qlabel[i].drawing_layer)

            if self.eraser_mode:
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.setPen(
                    QPen(Qt.transparent, pen_size + 2, Qt.SolidLine, Qt.RoundCap)
                )
            else:
                # smooth
                painter.setRenderHint(QPainter.Antialiasing)
                # Set the composition mode to control alpha blending
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                painter.setPen(
                    QPen(self.pen_color, pen_size, Qt.SolidLine, Qt.RoundCap)
                )

            painter.drawLine(self.paint_pos, event.pos())

            self.img_qlabel[i].update()  # schedule a repaint

        self.paint_pos = event.pos()  # update paint pos

    def draw_on_4_qlabels_release(self):
        self.paint_pos = None

    def __confirm_annotation(self):
        if self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER:
            # add clicks into 3d img
            pos = self.clicks_pos[0]
            self._3d_imgs["gtvt.click"][pos[0]][pos[1]][pos[2]] = 1
            print(self._3d_imgs["gtvt.click"].max())

            # save gtvt_clicks
            round_dir = os.path.join(
                g.TRAIN_RESULTS_DIR,
                self._baseline_id,
                self._idl_id["gtvt"],
                "patients",
                "patient={}".format(self._cur_patient),
                "round=01",
            )
            Dir.create(round_dir)
            idl_gtvt_click = self._3d_imgs["gtvt.click"].copy()
            # flip left/right for 1mm data
            if self._nii_spacing[2] == 1.0:
                idl_gtvt_click = np.flip(idl_gtvt_click, axis=2)
            # turn upside down
            idl_gtvt_click = np.flip(idl_gtvt_click, axis=0)
            Nii.save(
                img=idl_gtvt_click,
                save_path=os.path.join(round_dir, "gtvt_click.nii"),
                spacing=self._nii_spacing,
            )

            # clear clicks positions
            self.clicks_pos = List()

            self.set_cur_patient_idl_step(IDLStep.DRAW_GTVT)
            self._refresh_rgb_imgs()
            self._refresh_title()
            self.__refresh_crosses_on_rgb_imgs()
            self.__save_idl_step()
            self.__update_annotation_msg()
            print(self._3d_imgs["gtvt.click"].max())

        elif self.get_cur_patient_idl_step() == IDLStep.DRAW_GTVT:
            self.set_cur_patient_idl_step(IDLStep.CLICK_GTVN_CENTER)
            self.__clear_annotation()
            self._refresh_rgb_imgs()
            self._refresh_title()
            self.__save_idl_step()
            self.__update_annotation_msg()

        elif self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER:
            # add clicks into 3d img
            for pos in self.clicks_pos:
                self._3d_imgs["gtvn.clicks"][pos[0]][pos[1]][pos[2]] = 1
            # clear clicks positions
            self.clicks_pos = List()

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
            # update idl step for current patient
            self.set_cur_patient_idl_step(IDLStep.CORRECTION)

            self._choose_idl_gtvt()
            self._choose_idl_gtvn()
            self._refresh_rgb_imgs()
            self._refresh_title()
            self.__refresh_crosses_on_rgb_imgs()
            self.__save_idl_step()
            self.__update_annotation_msg()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.__refresh_crosses_on_rgb_imgs()

    def _set_img_plane(self):
        super()._set_img_plane()
        self.__refresh_crosses_on_rgb_imgs()

    def __refresh_crosses_on_rgb_imgs(self):
        # remove old crosses
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].delete_all_crosses()

        # load crosses position from self.clicks_pos
        if (
            self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVT_CENTER
            or self.get_cur_patient_idl_step() == IDLStep.CLICK_GTVN_CENTER
        ):
            img_shape = self.get_3d_img_shape()

            for d, h, w in self.clicks_pos:
                x = y = None
                if self._plane == Plane.TRANSVERSE:
                    if self._cur_slice == d:
                        x = w / img_shape[2]
                        y = h / img_shape[1]

                elif self._plane == Plane.CORONAL:
                    if self._cur_slice == h:
                        x = w / img_shape[2]
                        y = d / img_shape[0]

                elif self._plane == Plane.SAGITTAL:
                    if self._cur_slice == w:
                        x = h / img_shape[1]
                        y = d / img_shape[0]

                # find click on current slice
                if x is not None and y is not None:
                    x *= self._rgb_img_relative_pos["width"]
                    y *= self._rgb_img_relative_pos["height"]
                    x = round(x)
                    y = round(y)
                    x += self._rgb_img_relative_pos["x"]
                    y += self._rgb_img_relative_pos["y"]

                    # do not record click pos when refreshing
                    self.add_4_crosses(QPoint(x, y), record_click_pos=False)

    def delete_click_in_3d(self, cross: DraggableCross):
        pos = cross.get_pos_in_3d()
        self.clicks_pos.remove(pos)

    def add_click_in_3d(self, cross: DraggableCross):
        pos = cross.get_pos_in_3d()
        self.clicks_pos.append(pos)

    def set_4_crosses_dragging_offset(self, pos: QPoint):
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].selected_cross.offset = pos

    def set_4_crosses_dragging_state(self, dragging: bool):
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].selected_cross.dragging = dragging

    def move_4_crosses(self, pos: QPoint):
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].selected_cross.move(pos)

    def delete_4_crosses(self):
        cross = self.img_qlabel["ct"].selected_cross
        self.delete_click_in_3d(cross)
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].delete_selected_cross()

    # make this function public, CustomQLabel will use it
    def add_4_crosses(self, pos: QPoint, record_click_pos: bool):
        if self._3d_imgs["ct"] is None:
            return

        # make sure new cross id is unique
        crosses_id_list = self.img_qlabel["ct"].get_crosses_id_list()
        while 1:
            cross_id = random.randint(0, 2**16)
            if cross_id not in crosses_id_list:
                break
        # add crosses
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].add_cross(pos=pos, cross_id=cross_id)

        # add clicks into 3d img
        if record_click_pos:
            new_cross = self.img_qlabel["ct"].get_cross_by_id(cross_id)
            pos = new_cross.get_pos_in_3d()
            self.clicks_pos.append(pos)

    def select_4_crosses(self, cross_id: int):
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].select_cross(cross_id)

    def get_rgb_img_relative_pos(self):
        return self._rgb_img_relative_pos

    def get_nii_spacing(self):
        return self._nii_spacing

    def get_img_plane(self):
        return self._plane

    def get_cur_slice(self):
        return self._cur_slice

    def get_3d_img_shape(self):
        if self._3d_imgs["ct"] is not None:
            return self._3d_imgs["ct"].shape
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
        self.__btn["drawing.mode"] = self._btn_drawing_mode
        self.__btn["clear"] = self._btn_clear
        self.__btn["confirm"] = self._btn_confirm

    def _init_member_var(self, idl_remark: str = None, debug_mode: bool = False):
        super()._init_member_var()

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

        # save the position of gtvt/gtvn clicks
        self.clicks_pos = List()

        # drawing
        self.eraser_mode = False
        self.paint_pos = None  # Store the last painted point

        # r/g/b/transparency, all range from 0 to 255
        self.pen_color = Qt.green  # QColor(0, 255, 0, 150)

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

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def __clear_annotation(self):
        # clear drawing layer of each img_qlabel
        for i in ["ct", "pt", "mr1", "mr2"]:
            self.img_qlabel[i].drawing_layer = QPixmap(self.size())
            self.img_qlabel[i].drawing_layer.fill(Qt.transparent)
            self.img_qlabel[i].drawing_layer = self.img_qlabel[i].drawing_layer.scaled(
                self.img_qlabel[i].size()
            )
            self.img_qlabel[i].update()

    def __switch_drawing_mode(self):
        if self.get_cur_patient_idl_step() == IDLStep.DRAW_GTVT:
            if self.eraser_mode:
                self.eraser_mode = False
                icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "eraser.png"))
                self.__btn["drawing.mode"].setIcon(icon)
            else:
                self.eraser_mode = True
                icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "pen.png"))
                self.__btn["drawing.mode"].setIcon(icon)

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
        self._progress_bar_idl.show()
        self._slider_pen_size.show()
        for i in ["annotation.tools", "idl.progress", "pen.size"]:
            self._text_label[i].show()
        for i in ["drawing.mode", "clear", "confirm"]:
            self.__btn[i].show()

        # set text
        self._text_box_annotation_msg.setText("Please Select a Patient")
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

        # set icons
        icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "eraser.png"))
        self.__btn["drawing.mode"].setIcon(icon)
        icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "clear.png"))
        self.__btn["clear"].setIcon(icon)
        icon = QIcon(os.path.join(g.PROJ_DIR, "icons", "confirm.png"))
        self.__btn["confirm"].setIcon(icon)

        # connect ui to functions
        # (put this at the end, because these functions will need the initialization above)
        self.__btn["drawing.mode"].clicked.connect(self.__switch_drawing_mode)
        self.__btn["clear"].clicked.connect(self.__clear_annotation)
        self.__btn["confirm"].clicked.connect(self.__confirm_annotation)

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
        annotation_btn_width = 60

        # annotation tools
        top += gap
        rect = QRect(left, top, width, text_height)
        self._text_label["annotation.tools"].setGeometry(rect)
        self._text_label["annotation.tools"].show()
        top += text_height
        tmp_left = left
        annotation_btn_gap = round((width - 3 * annotation_btn_width) / 2)
        for i in ["drawing.mode", "clear", "confirm"]:
            rect = QRect(tmp_left, top, annotation_btn_width, bar_height)
            self.__btn[i].setGeometry(rect)
            self.__btn[i].show()
            tmp_left += annotation_btn_gap + annotation_btn_width
        top += bar_height

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

    def _choose_baseline(self):
        # self._reset_zoomin()
        self._clear_img_data()
        self._clear_img_qlabels()

        self._baseline_id = "baseline_real.idl"
        # self._baseline_id = "baseline_2023.02.27.07.08.09_3mm"

        # fill combobox patient after self._baseline_id is confirmed
        self._fill_combox_patient()
        self._combox["patient"].setCurrentIndex(-1)  # show nothing

        # # run this after patient combox current text is set up
        # self._enable_arrow_btns("patient")

        # create idl folders (after baseline_id is confirmed)
        for i in ["gtvt", "gtvn"]:
            Dir.create(
                os.path.join(g.TRAIN_RESULTS_DIR, self._baseline_id, self._idl_id[i])
            )

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

        for i in ["t", "n"]:
            if self._3d_imgs["gtv{}.pred".format(i)] is not None:
                cv_text = "GTV{}".format(i)
                pos_y += gap_y
                self._cv_put_text(
                    img=rgb_img,
                    text=cv_text,
                    pos=(pos_x, pos_y),
                    color=self._color["gtv{}.pred".format(i)],
                )

    def _choose_patient(self, idx: int = None):
        self._cur_patient = self._combox["patient"].currentText()
        # run these after patient combox current text is set up
        self._enable_arrow_btns("patient")
        self._load_dataset_dir_and_nii_spacing()

        # self._reset_zoomin()

        # load multi-modal imgs only, no labels
        self._load_multi_modal_imgs()
        # reset current slice id after ct img loaded
        self._reset_cur_slice_id()
        self._choose_idl_gtvt()
        self._choose_idl_gtvn()
        self._refresh_rgb_imgs()
        self._refresh_title()
        self.__refresh_crosses_on_rgb_imgs()
        self.__save_idl_step()
        self.__update_annotation_msg()

    def _reset_cur_slice_id(self):
        self._cur_slice = self._get_middle_slice_id()

    def get_cur_patient_idl_step(self):
        return self.__idl_step["patient={}".format(self._cur_patient)]

    def set_cur_patient_idl_step(self, step: str):
        self.__idl_step["patient={}".format(self._cur_patient)] = step

    def __update_annotation_msg(self):
        cur_patient_idl_step = self.get_cur_patient_idl_step()

        if cur_patient_idl_step == IDLStep.CLICK_GTVT_CENTER:
            self._text_box_annotation_msg.setText(
                "Please click the center of GTVt, then press OK"
            )

        elif cur_patient_idl_step == IDLStep.DRAW_GTVT:
            self._text_box_annotation_msg.setText(
                "Please delineate the countour of GTVt on transvers/coronal/sagittal plane, then press OK"
            )

        elif cur_patient_idl_step == IDLStep.CLICK_GTVN_CENTER:
            self._text_box_annotation_msg.setText(
                "Please click the center of each involved lymph nodes, then press OK."
            )

        elif cur_patient_idl_step == IDLStep.CORRECTION:
            self._text_box_annotation_msg.setText(
                "Please correct the predictions, then press OK"
            )

        else:
            Debug.error_exit("idl step value error")

    def _choose_idl_gtvt(self):
        patient_dir = self.__choose_idl(gtv="gtvt")
        # load gtvt click
        gtvt_click_nii_path = os.path.join(patient_dir, "round=01", "gtvt_click.nii")
        if os.path.exists(gtvt_click_nii_path):
            self._3d_imgs["gtvt.click"] = self._load_3d_img(
                path=gtvt_click_nii_path, binary=True
            )
        else:
            self._3d_imgs["gtvt.click"] = np.zeros(
                self._3d_imgs["ct"].shape, dtype=np.float32
            )

    def _choose_idl_gtvn(self):
        patient_dir = self.__choose_idl(gtv="gtvn")
        # load gtvn click
        gtvn_clicks_nii_path = os.path.join(patient_dir, "round=01", "gtvn_clicks.nii")
        if os.path.exists(gtvn_clicks_nii_path):
            self._3d_imgs["gtvn.clicks"] = self._load_3d_img(
                path=gtvn_clicks_nii_path, binary=True
            )
        else:
            self._3d_imgs["gtvn.clicks"] = np.zeros(
                self._3d_imgs["ct"].shape, dtype=np.float32
            )

    def __choose_idl(self, gtv: str) -> str:
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
                pred_path = os.path.join(round_dir, "{}_pred.nii".format(gtv))

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
