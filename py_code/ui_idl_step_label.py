from custom import Debug
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget
from str_lib import IDLStep


class LabelStatus:
    DONE = "\u2714"
    ONGOING = "\u25B6"
    NOT_START = ""
    MISSING = "\u2716"  # this is only for gtvt transverse/coronal/sagittal delineation


class IDLStepLabel(QLabel):
    def __init__(self, idl_step: str, parent: QWidget = None):
        super().__init__(parent=parent)
        self.__idl_step = idl_step
        self.__status = LabelStatus.NOT_START
        self.__font_weight = "light"
        self.__text_color = "white"
        self.__bg_color = "gray"

        # init text
        str_space = "            "
        if self.__idl_step == IDLStep.SELECT_PATIENT:
            self.setText("STEP 1 - Select a Patient")
        if self.__idl_step == IDLStep.CLICK_GTVT_CENTER:
            self.setText("STEP 2 - Click GTVt center")
        if self.__idl_step == IDLStep.DRAW_GTVT:
            self.setText("STEP 3 - Delineate GTVt")
        if self.__idl_step == IDLStep.DRAW_GTVT_TRANSVERSE:
            self.setText(str_space + "- in Transverse")
        if self.__idl_step == IDLStep.DRAW_GTVT_CORONAL:
            self.setText(str_space + "- in Coronal")
        if self.__idl_step == IDLStep.DRAW_GTVT_SAGITTAL:
            self.setText(str_space + "- in Sagittal")
        if self.__idl_step == IDLStep.CLICK_GTVN_CENTER:
            self.setText("STEP 4 - Click GTVn center")
        if self.__idl_step == IDLStep.WAITING:
            self.setText("STEP 5 - Generating Segmentation")
        if self.__idl_step == IDLStep.CORRECT_GTVT:
            self.setText("STEP 6 - Correct GTVt Segmentation")
        if self.__idl_step == IDLStep.CORRECT_GTVN:
            self.setText(str_space + "- Correct GTVn Segmentation")

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.window().on_idl_step_text_box_clicked(self)

    def __refresh_style(self):
        if self.__status in [LabelStatus.ONGOING, LabelStatus.MISSING]:
            self.setToolTip("Currently at this step")
        elif self.__idl_step in [
            IDLStep.SELECT_PATIENT,
            IDLStep.CLICK_GTVT_CENTER,
            IDLStep.DRAW_GTVT,
            IDLStep.CLICK_GTVN_CENTER,
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
            Debug.error_exit("Invalid input_status value!")
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
