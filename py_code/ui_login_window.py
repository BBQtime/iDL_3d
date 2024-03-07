import os
import sys

import qdarktheme
from custom import GPU, Debug, Dict, Dir
from custom import Global as g
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from ui_custom_combox import CustomComboBox
from ui_idl_window import IDLWindow
from ui_replay_window import ReplayWindow


class LoginWindow(QtWidgets.QMainWindow):
    def __center(self):
        # Get the application instance
        app = QApplication.instance()
        # Get the primary screen
        primary_screen = app.primaryScreen()
        # Get the geometry of the primary screen
        rect = primary_screen.availableGeometry()
        # Calculate the center position
        center_point = rect.center()
        # Adjust the window position
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(center_point)
        # Move the window to the center of the screen
        self.move(frame_geometry.topLeft())

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        width = 400 if g.is_linux() else 600
        height = 200 if g.is_linux() else 300
        self.resize(width, height)
        self.__center()
        self.setStyleSheet(g.FONT_STYLE)

        # Initialize combo boxes and text labels
        text_label = Dict()
        self.__combox = Dict()
        for i in ["user.name", "train.id"]:
            text_label[i] = QtWidgets.QLabel()
            self.__combox[i] = CustomComboBox()
            height = 27 if g.is_linux() else 40
            self.__combox[i].setFixedHeight(height)
        text_label["user.name"].setText("User")
        text_label["train.id"].setText("Experiment ID")

        # Layout
        sub_layout = Dict()
        layout_container = Dict()
        for i in ["user.name", "train.id"]:
            sub_layout[i] = QtWidgets.QVBoxLayout()
            sub_layout[i].addWidget(text_label[i])
            sub_layout[i].addWidget(self.__combox[i])
            sub_layout[i].setSpacing(1)
            layout_container[i] = QtWidgets.QWidget()
            layout_container[i].setLayout(sub_layout[i])

        # Central widget
        v_layout = QtWidgets.QVBoxLayout()
        for i in ["user.name", "train.id"]:
            v_layout.addWidget(layout_container[i])
        v_layout.setSpacing(30)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(v_layout)
        self.setCentralWidget(central_widget)

        # Add items to the combo boxes
        self.__combox["user.name"].addItems(
            [
                "~ Administrator",
                "Hanna Rahbek Mortensen",
                "Kenneth Jensen",
                "Jesper Grau Eriksen",
            ]
        )
        self.__combox["user.name"].sort()
        self.__combox["user.name"].setCurrentIndex(-1)

        # Connect the combo boxes to the function
        self.__combox["user.name"].activated.connect(self.__fill_combox_results)
        self.__combox["train.id"].activated.connect(self.__open_main_window)

    def __simplify_user_name(self):
        user_name = self.__combox["user.name"].currentText()
        for i in ["Admin", "Hanna", "Kenneth", "Jesper"]:
            if i in user_name:
                user_name = i
                break
        return user_name

    def __fill_combox_results(self):
        self.__combox["train.id"].clear()
        self.__combox["train.id"].addItem("Start a new experiment")

        user_name = self.__simplify_user_name()

        # get all train results
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_real.idl")
        if user_name == "Admin":
            key_word = ""
        else:
            key_word = user_name
        train_id_list = Dir.get_sub_dirs(input_dir=baseline_dir, key_word=key_word)

        # add train results into combobox
        for train_id in train_id_list:
            if train_id.startswith("idl.gtvt"):
                train_id = train_id[len("idl.gtvt_") :]
                self.__combox["train.id"].addItem(train_id)

        # add replay mode for administrator
        if user_name == "Admin":
            self.__combox["train.id"].addItem("Replay Mode")

        self.__combox["train.id"].setCurrentIndex(-1)

    def __open_main_window(self):
        # close the login window
        self.close()

        train_id = self.__combox["train.id"].currentText()

        # Open the main window
        if train_id == "Replay Mode":
            self.__main_window = ReplayWindow()
        else:
            self.__main_window = IDLWindow(
                user_name=self.__simplify_user_name(),
                train_id=train_id,
            )

        # install the event filter on the QApplication instance
        # This ensures that key press events will always trigger the main window's event handler,
        # regardless of which widget currently has focus.
        app = QApplication.instance()
        app.installEventFilter(self.__main_window)


# clear cache
if 0:
    GPU.clear_cache()
    Debug.clear_debug_data()
    Debug.clear_linux_trash()


# show login window
app = QApplication(sys.argv)
qdarktheme.setup_theme()
login_window = LoginWindow()
login_window.show()
sys.exit(app.exec_())
