from custom import Debug
from PyQt5.QtWidgets import QLabel, QWidget


class TextBoxStatus:
    DONE = "\u2714"
    ONGOING = "\u25B6"
    NOT_START = ""
    MISSING = "\u2716"


class TextBox(QLabel):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent=parent)
        # self.setEnabled(False)
        self.__status = TextBoxStatus.NOT_START
        self.__font_weight = "light"
        self.__text_color = "white"
        self.__bg_color = "gray"
        self.refresh_style()

    def refresh_style(self):
        if self.__status == TextBoxStatus.DONE:
            self.__font_weight = "light"
            self.__text_color = "white"
            self.__bg_color = "green"
        elif self.__status == TextBoxStatus.ONGOING:
            self.__font_weight = "bold"
            self.__text_color = "white"
            self.__bg_color = "orange"
        elif self.__status == TextBoxStatus.NOT_START:
            self.__font_weight = "light"
            self.__text_color = "white"
            self.__bg_color = "gray"
        elif self.__status == TextBoxStatus.MISSING:
            self.__font_weight = "bold"
            self.__text_color = "white"
            self.__bg_color = "red"

        # update style
        style = """
        font-weight: {font_weight};
        border: 2px solid dark-gray;
        color: {text_color};
        background-color: {bg_color};
        """
        style = style.format(
            font_weight=self.__font_weight,
            text_color=self.__text_color,
            bg_color=self.__bg_color,
        )
        self.setStyleSheet(style)

    # def set_text_color(self, text_color: str):
    #     self.__text_color = text_color
    #     self.refresh_style()

    # def set_bg_color(self, bg_color: str):
    #     self.__bg_color = bg_color
    #     self.refresh_style()

    def __remove_head_symbol(self):
        for head_symbol in [
            TextBoxStatus.DONE,
            TextBoxStatus.ONGOING,
            TextBoxStatus.MISSING,
        ]:
            if head_symbol in self.text():
                text = self.text()
                # Remove the done_symbol from the string
                text = text.replace(head_symbol, "")
                super().setText(text)

    def setText(self, text: str):
        super().setText(self.__status + text)
        self.refresh_style()

    def __set_status(self, input_status: str):
        if input_status not in [
            TextBoxStatus.DONE,
            TextBoxStatus.ONGOING,
            TextBoxStatus.NOT_START,
            TextBoxStatus.MISSING,
        ]:
            Debug.error_exit("Invalid input_status value!")
        self.__remove_head_symbol()
        text = self.text()
        self.__status = input_status
        self.setText(text)
        self.refresh_style()

    def set_status_done(self):
        self.__set_status(TextBoxStatus.DONE)

    def set_status_ongoing(self):
        self.__set_status(TextBoxStatus.ONGOING)

    def set_status_notstart(self):
        self.__set_status(TextBoxStatus.NOT_START)

    def set_status_missing(self):
        self.__set_status(TextBoxStatus.MISSING)
