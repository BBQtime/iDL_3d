from PyQt5.QtWidgets import QPushButton


class ToggleButton(QPushButton):
    def __init__(self, parent=None, is_checked: bool = False):
        super().__init__("", parent)
        self.setCheckable(True)
        self.setStyleSheet("QPushButton {border: none; padding: 0;}")
        self.setFixedSize(32, 32)
        self.setChecked(is_checked)
        self.__update_style(self.isChecked())
        self.clicked.connect(self.on_toggle)

    def on_toggle(self):
        self.__update_style(self.isChecked())
        self.window().switch_display_mode()

    def __update_style(self, is_checked):
        if is_checked:
            self.setStyleSheet(
                "QPushButton{border-image:url(/mnt/faststorage/alan/iDL_3d/icons/toggle_right.png)}"
            )
        else:
            self.setStyleSheet(
                "QPushButton{border-image:url(/mnt/faststorage/alan/iDL_3d/icons/toggle_left.png)}"
            )
