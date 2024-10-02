import sys

import qdarktheme
from global_utils import global_core as g
from PyQt5.QtWidgets import QApplication
from ui_utils.login_window import LoginWindow

if __name__ == "__main__":
    g.clear_gpu_cache()
    g.clear_linux_trash()
    # g.clear_debug_data()

    # show login window
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    login_window = LoginWindow()
    # Apply event filter to login window, block key press event and mouse wheel event
    app.installEventFilter(login_window)
    login_window.show()
    sys.exit(app.exec_())
