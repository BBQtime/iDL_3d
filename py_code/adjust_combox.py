from PyQt5.QtWidgets import QComboBox


class AdjustableComboBox(QComboBox):
    # def showPopup(self):
    #     # Calculate the width of the largest item
    #     width = self.width()
    #     for i in range(self.count()):
    #         width = max(width, self.view().visualRect(self.model().index(i, 0)).width())

    #     self.view().setMinimumWidth(width)
    #     super().showPopup()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
