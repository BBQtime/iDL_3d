import global_utils.global_core as g
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget


class LabelStatus:
    COMPLETED = "\u2714"
    ACTIVE = "\u25B6"
    NOT_START = ""

    # red, this is only for gtvt transverse/coronal/sagittal delineation
    MISSING = "\u2716"


class TodoListLabel(QLabel):
    # key names for dict
    SELECT_PATIENT = "select.patient"
    CLICK_GTVT_CENTER = "click.gtvt.center"
    DELINEATE_GTVT = "delineate.gtvt"
    DELINEATE_GTVT_TRANSVERSE = "delineate.gtvt.transverse"
    DELINEATE_GTVT_CORONAL = "delineate.gtvt.coronal"
    DELINEATE_GTVT_SAGITTAL = "delineate.gtvt.sagittal"
    CLICK_GTVN_CENTERS = "click.gtvn.centers"
    WAIT_GTVT_PRED = "wait.gtvt.pred"
    WAIT_GTVN_PRED = "wait.gtvn.pred"
    CORRECT_GTVT = "correct.gtvt"
    CORRECT_GTVN = "correct.gtvn"

    def __init__(self, name: str, parent: QWidget = None):
        super().__init__(parent=parent)
        self.__name = name
        self.__status = LabelStatus.NOT_START
        self.__font_weight = "light"
        self.__text_color = "white"
        self.__bg_color = "gray"
        self.setFixedHeight(g.TEXT_HEIGHT + 3)

        # init text
        str_space = "            "
        if self.__name == TodoListLabel.SELECT_PATIENT:
            self.setText("STEP 1 - Select Patient")
        elif self.__name == TodoListLabel.CLICK_GTVT_CENTER:
            self.setText("STEP 2 - Click GTVt center")
        elif self.__name == TodoListLabel.DELINEATE_GTVT:
            self.setText("STEP 3 - Delineate GTVt")
        elif self.__name == TodoListLabel.DELINEATE_GTVT_TRANSVERSE:
            self.setText(str_space + "- in Transverse")
        elif self.__name == TodoListLabel.DELINEATE_GTVT_CORONAL:
            self.setText(str_space + "- in Coronal")
        elif self.__name == TodoListLabel.DELINEATE_GTVT_SAGITTAL:
            self.setText(str_space + "- in Sagittal")
        elif self.__name == TodoListLabel.CLICK_GTVN_CENTERS:
            self.setText("STEP 4 - Click GTVn center")
        elif self.__name == TodoListLabel.WAIT_GTVT_PRED:
            self.setText("STEP 5 - Generating GTVt Results")
        elif self.__name == TodoListLabel.WAIT_GTVN_PRED:
            self.setText(str_space + "- Generating GTVn Results")
        elif self.__name == TodoListLabel.CORRECT_GTVT:
            self.setText("STEP 6 - Correct GTVt")
        elif self.__name == TodoListLabel.CORRECT_GTVN:
            self.setText(str_space + "- Correct GTVn")

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.window().on_todo_list_clicked(self)

    def __refresh_style(self):
        if self.__status in [LabelStatus.ACTIVE, LabelStatus.MISSING]:
            self.setToolTip("Currently at this step")
        elif self.__name in [
            TodoListLabel.SELECT_PATIENT,
            TodoListLabel.CLICK_GTVT_CENTER,
            TodoListLabel.DELINEATE_GTVT,
            TodoListLabel.DELINEATE_GTVT_TRANSVERSE,
            TodoListLabel.DELINEATE_GTVT_CORONAL,
            TodoListLabel.DELINEATE_GTVT_SAGITTAL,
            TodoListLabel.CLICK_GTVN_CENTERS,
            TodoListLabel.CORRECT_GTVT,
            TodoListLabel.CORRECT_GTVN,
        ]:
            if self.__status == LabelStatus.COMPLETED:
                self.setToolTip("Click to revert to this step")
            elif self.__status == LabelStatus.NOT_START:
                self.setToolTip("")  # "CAN NOT jump to this step right now"
        # current step doesnt support jumping
        else:
            self.setToolTip("")  # "CAN NOT jump to this step"

        # set font and text/background/color
        if self.__status == LabelStatus.COMPLETED:
            self.__font_weight = "light"
            self.__text_color = "white"
            self.__bg_color = "green"
        elif self.__status == LabelStatus.ACTIVE:
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
            LabelStatus.COMPLETED,
            LabelStatus.ACTIVE,
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
            LabelStatus.COMPLETED,
            LabelStatus.ACTIVE,
            LabelStatus.NOT_START,
            LabelStatus.MISSING,
        ]:
            g.error_exit("Invalid input_status value!")
        self.__remove_head_symbol()
        text = self.text()
        self.__status = input_status
        self.setText(text)
        self.__refresh_style()

    def set_status_completed(self):
        self.__set_status(LabelStatus.COMPLETED)

    def set_status_active(self):
        self.__set_status(LabelStatus.ACTIVE)

    def set_status_not_start(self):
        self.__set_status(LabelStatus.NOT_START)

    def set_status_missing(self):
        self.__set_status(LabelStatus.MISSING)
