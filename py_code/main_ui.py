import sys

import qdarktheme
from PyQt5.QtWidgets import QApplication
from ui_utils.login_window import LoginWindow

if __name__ == "__main__":

    # show login window
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    login_window = LoginWindow()
    # Apply event filter to login window, block key press event and mouse wheel event
    app.installEventFilter(login_window)
    login_window.show()
    sys.exit(app.exec_())
