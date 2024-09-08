import global_utils.global_core as g
from global_utils.str_lib import ObsStudyStep
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget


class LabelStatus:
    DONE = "\u2714"
    ONGOING = "\u25B6"
    NOT_START = ""
    MISSING = "\u2716"  # this is only for gtvt transverse/coronal/sagittal delineation


class ObsStudyStepLabel(QLabel):
    def __init__(self, obs_study_step: str, parent: QWidget = None):
        super().__init__(parent=parent)
        self.__obs_study_step = obs_study_step
        self.__status = LabelStatus.NOT_START
        self.__font_weight = "light"
        self.__text_color = "white"
        self.__bg_color = "gray"
        self.setFixedHeight(g.TEXT_HEIGHT + 3)

        # init text
        str_space = "            "
        if self.__obs_study_step == ObsStudyStep.SELECT_PATIENT:
            self.setText("STEP 1 - Select Patient")
        elif self.__obs_study_step == ObsStudyStep.CLICK_GTVT_CENTER:
            self.setText("STEP 2 - Click GTVt center")
        elif self.__obs_study_step == ObsStudyStep.DRAW_GTVT:
            self.setText("STEP 3 - Delineate GTVt")
        elif self.__obs_study_step == ObsStudyStep.DRAW_GTVT_TRANSVERSE:
            self.setText(str_space + "- in Transverse")
        elif self.__obs_study_step == ObsStudyStep.DRAW_GTVT_CORONAL:
            self.setText(str_space + "- in Coronal")
        elif self.__obs_study_step == ObsStudyStep.DRAW_GTVT_SAGITTAL:
            self.setText(str_space + "- in Sagittal")
        elif self.__obs_study_step == ObsStudyStep.CLICK_GTVN_CENTER:
            self.setText("STEP 4 - Click GTVn center")
        elif self.__obs_study_step == ObsStudyStep.WAITING:
            self.setText("STEP 5 - Generating Results")
        elif self.__obs_study_step == ObsStudyStep.CORRECT_GTVT:
            self.setText("STEP 6 - Correct GTVt")
        elif self.__obs_study_step == ObsStudyStep.CORRECT_GTVN:
            self.setText(str_space + "- Correct GTVn")

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.window().on_idl_step_text_box_clicked(self)

    def __refresh_style(self):
        if self.__status in [LabelStatus.ONGOING, LabelStatus.MISSING]:
            self.setToolTip("Currently at this step")
        elif self.__obs_study_step in [
            ObsStudyStep.SELECT_PATIENT,
            ObsStudyStep.CLICK_GTVT_CENTER,
            ObsStudyStep.DRAW_GTVT,
            ObsStudyStep.DRAW_GTVT_TRANSVERSE,
            ObsStudyStep.DRAW_GTVT_CORONAL,
            ObsStudyStep.DRAW_GTVT_SAGITTAL,
            ObsStudyStep.CLICK_GTVN_CENTER,
            ObsStudyStep.CORRECT_GTVT,
            ObsStudyStep.CORRECT_GTVN,
        ]:
            if self.__status == LabelStatus.DONE:
                self.setToolTip("Click to revert to this step")
            elif self.__status == LabelStatus.NOT_START:
                self.setToolTip("")  # "CAN NOT jump to this step right now"
        # current step doesnt support jumping
        else:
            self.setToolTip("")  # "CAN NOT jump to this step"

        # set font and text/background/color
        if self.__status == LabelStatus.DONE:
            self.__font_weight = "light"
            self.__text_color = "white"
            self.__bg_color = "green"
        elif self.__status == LabelStatus.ONGOING:
            self.__font_weight = "bold"
            self.__text_color = "white"
            self.__bg_color = "orange"
        elif self.__status == LabelStatus.NOT_START:
            self.__font_weight = "light"
            self.__text_color = "white"
            self.__bg_color = "gray"
        elif self.__status == LabelStatus.MISSING:
            self.__font_weight = "bold"
            self.__text_color = "white"
            self.__bg_color = "orange"

        # update style
        style = """
        QLabel {{
            font-weight: {font_weight};
            border: 2px solid dark-gray;
            color: {text_color};
            background-color: {bg_color};
        }}
        QToolTip {{
            font-weight: light;
            color: dark-gray;
            background-color: #fff;
            border: 1px solid black;
        }}
        """
        style = style.format(
            font_weight=self.__font_weight,
            text_color=self.__text_color,
            bg_color=self.__bg_color,
        )
        self.setStyleSheet(style)

    def __remove_head_symbol(self):
        for head_symbol in [
            LabelStatus.DONE,
            LabelStatus.ONGOING,
            LabelStatus.MISSING,
        ]:
            if head_symbol in self.text():
                text = self.text()
                # Remove the done_symbol from the string
                text = text.replace(head_symbol, "")
                super().setText(text)

    def setText(self, text: str):
        super().setText(self.__status + text)

    def __set_status(self, input_status: str):
        if input_status not in [
            LabelStatus.DONE,
            LabelStatus.ONGOING,
            LabelStatus.NOT_START,
            LabelStatus.MISSING,
        ]:
            g.error_exit("Invalid input_status value!")
        self.__remove_head_symbol()
        text = self.text()
        self.__status = input_status
        self.setText(text)
        self.__refresh_style()

    def set_status_done(self):
        self.__set_status(LabelStatus.DONE)

    def set_status_ongoing(self):
        self.__set_status(LabelStatus.ONGOING)

    def set_status_notstart(self):
        self.__set_status(LabelStatus.NOT_START)

    def set_status_missing(self):
        self.__set_status(LabelStatus.MISSING)
