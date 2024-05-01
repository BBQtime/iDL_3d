import custom as g
from PyQt5.QtWidgets import QPushButton


class ToggleButton(QPushButton):
    def __init__(self, parent=None, is_checked: bool = False):
        super().__init__("", parent)
        self.setCheckable(True)
        style = """
        QPushButton {
            background-color: transparent;
            border: none;
            padding: 0;
        }
        QPushButton:pressed {
            background-color: transparent;
            border: none;
            padding: 0;
        }
        QPushButton:hover {
            background-color: transparent;
            border: none;
            padding: 0;
        }
        """
        self.setStyleSheet(style)
        btn_size = 32 if g.is_linux() else 38
        self.setFixedSize(btn_size, btn_size)
        self.setChecked(is_checked)
        self.__update_style(self.isChecked())
        self.clicked.connect(self.on_toggle)

    def on_toggle(self):
        self.__update_style(self.isChecked())
        self.window().switch_display_mode()

    def __update_style(self, is_checked):
        path = "QPushButton{border-image:url("

        if g.is_linux():
            path += "/mnt/faststorage/alan/"
        else:
            path += "E:/Alan/"

        path += "iDL_3d/icons/toggle_"

        if is_checked:
            path += "right"
        else:
            path += "left"

        path += ".png)}"
        self.setStyleSheet(path)
