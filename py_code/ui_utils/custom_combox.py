import global_utils.global_core as g
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, QWidget


class CustomItemDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        # Call the base class method to get the default size hint
        original_size = super().sizeHint(option, index)
        # Increase the height for more space between items
        gap = 8 if g.is_linux() else 10
        return QSize(original_size.width(), original_size.height() + gap)


class CustomComboBox(QComboBox):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        # set height of each elements in dropdown list
        self.setItemDelegate(CustomItemDelegate())

    def showPopup(self):
        super().showPopup()

        # Calculate the width of the largest item
        width = self.width()
        for i in range(self.count()):
            width = max(width, self.view().visualRect(self.model().index(i, 0)).width())
        self.view().setMinimumWidth(width)

        # alignment right
        # Calculate the width difference between the dropdown and the combobox
        width_difference = self.view().minimumWidth() - self.width()
        if width_difference > 0:
            # Get the current popup geometry
            popup_geometry = self.view().parentWidget().geometry()
            # Adjust the x position to align it to the right of the QComboBox
            popup_geometry.moveLeft(popup_geometry.x() - width_difference)
            # Set the new geometry for the popup
            self.view().parentWidget().setGeometry(popup_geometry)

    def sort(self):
        # Retrieve the items from the combobox
        items = [self.itemText(i) for i in range(self.count())]
        # Sort the items based on the first letter
        sorted_items = sorted(items, key=lambda item: item[0].lower())
        self.clear()
        self.addItems(sorted_items)
