import global_utils.global_core as g
from global_utils.str_lib import DisplayMode, DrawingMode, ObsStudyStep, Plane
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QLabel
from ui_utils.drag_cross import DragCross


class ImgFrame(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        # record plane and modality here instead of main window
        self.plane = None
        self.modal = None

        # right click drag img
        self.__dragging = False
        self.__drag_pos = None
        self.img_center_pct = (0.5, 0.5)

        # clicks
        self.selected_cross = None
        self.crosses_list = []

        # gtvt painting
        self.setMouseTracking(True)
        self.background_img = None
        self.drawing_layer = QPixmap(self.size())
        self.drawing_layer.fill(Qt.transparent)
        self.pen_mode = True

        # eraser circle
        self.__circle_pos = None

    def clear_drawing_layer(self):
        self.drawing_layer = QPixmap(self.size())
        self.drawing_layer.fill(Qt.transparent)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Resize the drawing layer pixmap to match the new size of the QLabel
        self.drawing_layer = self.drawing_layer.scaled(self.size())

    def mouse_press_event_left_button(self, event: QMouseEvent):
        if not self.window().is_obs_study_window():
            # update cur slices id
            pos_3d = self.get_pos_in_3d(event.pos())
            self.window().cur_slice_id[Plane.TRANSVERSE] = int(pos_3d[0])
            self.window().cur_slice_id[Plane.CORONAL] = int(pos_3d[1])
            self.window().cur_slice_id[Plane.SAGITTAL] = int(pos_3d[2])
            # in PLANE_FIXED mode, refresh other img_frames to switch to current slice
            if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                # (1) refresh other img_frames from scratch
                frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                frame_name_list.remove(self.plane)
                for i in frame_name_list:
                    self.window().refresh_imgs(frame_name=i)
                # (2) on current img frame, only refresh anatomical lines
                self.window().refresh_imgs(
                    frame_name=self.plane,
                    reload_origin_rgb=False,
                    reload_zoomed_rgb=False,
                    reload_contours=False,
                )

        elif self.window().obs_study_step is None:
            return

        elif self.window().obs_study_step == ObsStudyStep.CLICK_GTVT_CENTER:
            # remove old crosses
            self.window().delete_all_crosses()
            # add new 3d pos
            self.window().gtvt_click_pos_3d = self.get_pos_in_3d(event.pos())
            self.window().reset_cur_slice_id()
            # in PLANE_FIXED mode, refresh other img_frames to switch to current slice
            if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                # (1) refresh other img_frames from scratch
                frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                frame_name_list.remove(self.plane)
                for i in frame_name_list:
                    self.window().refresh_imgs(frame_name=i)
                # (2) on current img frame, only refresh anatomical lines
                self.window().refresh_imgs(
                    frame_name=self.plane,
                    reload_origin_rgb=False,
                    reload_zoomed_rgb=False,
                    reload_contours=False,
                )
            # refresh crosses
            self.window().refresh_crosses()

        # draw/correct
        elif self.window().obs_study_step in [
            ObsStudyStep.DRAW_GTVT,
            ObsStudyStep.CORRECT_GTVT,
            ObsStudyStep.CORRECT_GTVN,
            ObsStudyStep.CORRECT_BOTH,
        ]:
            self.window().draw_on_img_frame_press(event=event, img_frame=self)

        elif self.window().obs_study_step == ObsStudyStep.CLICK_GTVN_CENTER:
            pos_3d = self.get_pos_in_3d(event.pos())
            if pos_3d not in self.window().gtvn_clicks_pos_3d:
                self.window().gtvn_clicks_pos_3d.append(pos_3d)
                self.window().reset_cur_slice_id()
                # in PLANE_FIXED mode, refresh other img_frames to switch to current slice
                if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                    # (1) refresh other img_frames from scratch
                    frame_name_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                    frame_name_list.remove(self.plane)
                    for i in frame_name_list:
                        self.window().refresh_imgs(frame_name=i)
                    # (2) on current img frame, only refresh anatomical lines
                    self.window().refresh_imgs(
                        frame_name=self.plane,
                        reload_origin_rgb=False,
                        reload_zoomed_rgb=False,
                        reload_contours=False,
                    )
                # refresh crosses on all img frames
                self.window().refresh_crosses()

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            self.mouse_press_event_left_button(event)

        elif event.button() == Qt.RightButton:
            self.__dragging = True
            self.__drag_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)

        should_paint_eraser_circle = self.__should_paint_eraser_circle()

        # put this before draw_on_img_frame_move()
        # because draw_on_img_frame_move will trigger repaint
        if should_paint_eraser_circle:
            self.__circle_pos = event.pos()

        # in "mouseMoveEvent", use event.buttons() instead of event.button()
        # button() returns the mouse button that caused the event, which is Qt::NoButton
        if event.buttons() == Qt.LeftButton:
            # this function will trigger repaint
            self.window().draw_on_img_frame_move(event=event, img_frame=self)

        elif should_paint_eraser_circle:
            self.update()  # repaint

        # right click dragging
        if event.buttons() == Qt.RightButton and self.__dragging:
            diff_x = event.pos().x() - self.__drag_pos.x()
            diff_y = event.pos().y() - self.__drag_pos.y()
            img_pos_diff = (diff_x, diff_y)
            self.__drag_pos = event.pos()  # update offset

            # refresh img/imgs and crosses
            # (1) PLANE_FIXED mode: refresh current img frame
            if self.window().display_mode() == DisplayMode.PLANE_FIXED:
                # only update position
                # no need to reload origin_rgb, zoomed_rgb, contours
                self.window().refresh_imgs(
                    frame_name=self.plane,
                    reload_origin_rgb=False,
                    reload_zoomed_rgb=False,
                    reload_contours=False,
                    img_pos_diff=img_pos_diff,
                )
                # refresh crosses on self (current img frame)
                self.refresh_crosses()

            # (2) MODAL_FIXED mode: refresh all 4 img frames
            else:
                # only update position
                # no need to reload origin_rgb, zoomed_rgb, contours
                self.window().refresh_imgs(
                    reload_origin_rgb=False,
                    reload_zoomed_rgb=False,
                    reload_contours=False,
                    img_pos_diff=img_pos_diff,
                )
                # refresh crosses on all img frames
                self.window().refresh_crosses()

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton:
            self.window().draw_on_img_frame_release(self)

        elif event.button() == Qt.RightButton:
            self.__dragging = False

    def __should_paint_eraser_circle(self):
        if not self.window().is_obs_study_window():
            return False

        elif self.window().obs_study_step in [
            ObsStudyStep.DRAW_GTVT,
            ObsStudyStep.CORRECT_GTVT,
            ObsStudyStep.CORRECT_GTVN,
            ObsStudyStep.CORRECT_BOTH,
        ]:
            if self.window().drawing_mode in [
                DrawingMode.GTVT_ERASER,
                DrawingMode.GTVN_ERASER,
            ]:
                return True
            else:
                return False
        else:
            return False

    # This function is called when the mouse enters the QLabel area
    def enterEvent(self, event):
        super().enterEvent(event)

        self.window().change_mouse_cursor(check_mouse_over_img_frame=False)
        if self.__should_paint_eraser_circle():
            self.__circle_pos = event.pos()
            self.update()  # repaint
        else:
            self.__circle_pos = None

    # This function is called when the mouse leaves the QLabel area
    def leaveEvent(self, event):
        super().leaveEvent(event)

        self.window().restore_mouse_cursor()
        self.__circle_pos = None
        if self.__should_paint_eraser_circle():
            self.update()  # repaint

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.background_img:
            painter.setOpacity(1.0)
            painter.drawPixmap(self.rect(), self.background_img)

        if self.drawing_layer:
            # 0 for fully transparent, 255 for fully opaque
            # (1) pen transparency
            if self.pen_mode:
                painter.setOpacity(130 / 255)
            # (2) eraser transparency
            else:
                painter.setOpacity(180 / 255)
            painter.drawPixmap(self.rect(), self.drawing_layer)

        # draw eraser circle
        if self.__circle_pos and self.window().is_obs_study_window():
            # circle color
            # (1) delineate gtvt
            if self.window().obs_study_step == ObsStudyStep.DRAW_GTVT:
                circle_color = self.window().color["gtvt.delineation"]
            # (2) correct gtvt/gtvn
            else:
                if self.window().drawing_mode == DrawingMode.GTVT_ERASER:
                    circle_color = self.window().color["gtvt.pred"]
                elif self.window().drawing_mode == DrawingMode.GTVN_ERASER:
                    circle_color = self.window().color["gtvn.pred"]

            circle_color = QtGui.QColor(*circle_color)
            pen = QtGui.QPen(circle_color)
            # circle size
            circle_border_width = 3
            pen.setWidth(circle_border_width)
            # draw a non-transparent circle
            painter.setOpacity(1.0)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            radius = self.window().get_eraser_size() / 2.0
            # Draw circle at the mouse position
            painter.drawEllipse(self.__circle_pos, radius, radius)

    def set_background(self, img: QImage):
        self.background_img = QPixmap.fromImage(img)
        self.update()

    def get_cross_by_id(self, cross_id: int) -> DragCross:
        for cross in self.crosses_list:
            if cross.cross_id == cross_id:
                return cross
        # no id found, return None
        return None

    def get_crosses_id_list(self) -> list:
        crosses_id_list = []
        for cross in self.crosses_list:
            crosses_id_list.append(cross.cross_id)
        return crosses_id_list

    def deselect_cross(self):
        if self.selected_cross:
            self.selected_cross.select(False)
            self.selected_cross = None

    def select_cross(self, cross_id: tuple):
        self.deselect_cross()
        # select new cross
        for cross in self.crosses_list:
            if cross.cross_id == cross_id:
                self.selected_cross = cross
                self.selected_cross.select(True)

    def delete_all_crosses(self):
        for cross in self.crosses_list:
            cross.setParent(None)
            cross.deleteLater()
        self.crosses_list = []
        self.selected_cross = None

    def delete_selected_cross(self):
        if self.selected_cross:
            self.window().remove_3d_pos_of_selected_cross(self.selected_cross)
            self.crosses_list.remove(self.selected_cross)
            self.selected_cross.setParent(None)
            self.selected_cross.deleteLater()
            self.selected_cross = None

    def get_pos_in_3d(self, pos: QPoint):
        cur_slice = self.window().cur_slice_id[self.plane]
        img_shape_3d = self.window().get_3d_img_shape()

        center_x_pct, center_y_pct = self.img_center_pct
        frame_name = self.get_frame_name()
        zoomed_h, zoomed_w = self.window().get_zoomed_rgb_shape(frame_name)
        origin_h, origin_w = self.window().get_origin_rgb_shape(frame_name)

        # abs position in zoomed image
        center_x_zoom = int(round(zoomed_w * center_x_pct))
        center_y_zoom = int(round(zoomed_h * center_y_pct))
        # top left corner in zoomed image
        start_x_zoom = center_x_zoom - self.width() / 2
        start_y_zoom = center_y_zoom - self.height() / 2
        # position in zoomed rgb
        x_pct = (start_x_zoom + pos.x()) / zoomed_w
        y_pct = (start_y_zoom + pos.y()) / zoomed_h
        # position in origin rgb
        x = origin_w * x_pct
        y = origin_h * y_pct
        # float -> int
        x = round(x)
        y = round(y)

        # 2d to 3d
        d, h, w = img_shape_3d
        if self.plane == Plane.TRANSVERSE:
            w = x
            h = y
            d = cur_slice
        elif self.plane == Plane.CORONAL:
            w = x
            h = cur_slice
            d = y
        elif self.plane == Plane.SAGITTAL:
            w = cur_slice
            h = x
            d = y

        w = g.clamp_value(w, (0, img_shape_3d[2] - 1))
        h = g.clamp_value(h, (0, img_shape_3d[1] - 1))
        d = g.clamp_value(d, (0, img_shape_3d[0] - 1))

        # (1) dont neet to turn upside down

        # (2) flip left/right back
        # w = img_shape_3d[2] - w

        # (3) make sure transverse slice id is a multiple of interpolation step
        if self.window().is_obs_study_window():
            d = self.window().ensure_slice_id_multiple(
                slice_id=d,
                slice_count=img_shape_3d[0],
            )

        return d, h, w

    def get_frame_name(self):
        frame_name = None
        for i in self.window().img_frame.keys():
            if self.window().img_frame[i] == self:
                frame_name = i
        if frame_name is None:
            g.error_exit("img_frame error")
        else:
            return frame_name

    def refresh_crosses(self):
        if not self.window().is_obs_study_window():
            return

        if self.window().obs_study_step not in [
            ObsStudyStep.CLICK_GTVT_CENTER,
            ObsStudyStep.CLICK_GTVN_CENTER,
        ]:
            return

        # remove old crosses
        self.delete_all_crosses()

        # load crosses position from gtvt/gtvn_clicks_pos_3d
        if self.window().obs_study_step == ObsStudyStep.CLICK_GTVT_CENTER:
            if self.window().gtvt_click_pos_3d is None:
                return
            else:
                clicks_pos_3d = [self.window().gtvt_click_pos_3d]

        elif self.window().obs_study_step == ObsStudyStep.CLICK_GTVN_CENTER:
            if len(self.window().gtvn_clicks_pos_3d) <= 0:
                return
            else:
                clicks_pos_3d = self.window().gtvn_clicks_pos_3d

        # get zoomed in values
        img_shape_3d = self.window().get_3d_img_shape()
        center_x_pct, center_y_pct = self.img_center_pct
        frame_name = self.get_frame_name()
        zoomed_h, zoomed_w = self.window().get_zoomed_rgb_shape(frame_name)

        # loop through all clicks
        for d, h, w in clicks_pos_3d:
            x_pct = y_pct = None
            if (
                self.plane == Plane.TRANSVERSE
                and self.window().cur_slice_id[Plane.TRANSVERSE] == d
            ):
                x_pct = w / img_shape_3d[2]
                y_pct = h / img_shape_3d[1]

            elif (
                self.plane == Plane.CORONAL
                and self.window().cur_slice_id[Plane.CORONAL] == h
            ):
                x_pct = w / img_shape_3d[2]
                y_pct = d / img_shape_3d[0]

            elif (
                self.plane == Plane.SAGITTAL
                and self.window().cur_slice_id[Plane.SAGITTAL] == w
            ):
                x_pct = h / img_shape_3d[1]
                y_pct = d / img_shape_3d[0]

            # find click on current slice
            if x_pct is not None and y_pct is not None:
                x_frame = self.width() // 2 - (center_x_pct - x_pct) * zoomed_w
                x_frame = round(x_frame)
                y_frame = self.height() // 2 - (center_y_pct - y_pct) * zoomed_h
                y_frame = round(y_frame)

                if x_frame >= 0 and y_frame >= 0:
                    # do not record click pos when refreshing
                    self.add_cross(
                        pos=QPoint(x_frame, y_frame),
                        click_pos_3d=(d, h, w),
                    )

    def add_cross(self, pos: QPoint, click_pos_3d: tuple):
        self.deselect_cross()
        # create new cross (use click_pos_3d as cross_id)
        new_cross = DragCross(parent=self, cross_id=click_pos_3d)
        new_cross.setGeometry(
            pos.x() - round(new_cross.CROSS_SIZE / 2),
            pos.y() - round(new_cross.CROSS_SIZE / 2),
            new_cross.CROSS_SIZE,
            new_cross.CROSS_SIZE,
        )
        new_cross.load_png(new_cross.UNSELECTED_CROSS_ICON_PATH)
        new_cross.show()
        self.crosses_list.append(new_cross)
