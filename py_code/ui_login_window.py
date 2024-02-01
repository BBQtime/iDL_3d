import os
import sys

from custom import GPU, Debug, Dict, Dir
from custom import Global as g
from darktheme.widget_template import DarkPalette
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from ui_idl_window import IDLWindow
from ui_replay_window import ReplayWindow


class LoginWindow(QtWidgets.QMainWindow):
    def __center(self):
        # Get the application instance
        app = QtWidgets.QApplication.instance()
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

    def __sort_combox(self, combox: QtWidgets.QComboBox):
        # Retrieve the items from the QComboBox
        items = [combox.itemText(i) for i in range(combox.count())]
        # Sort the items based on the first letter
        sorted_items = sorted(items, key=lambda item: item[0].lower())
        combox.clear()
        combox.addItems(sorted_items)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.resize(400, 200)
        self.__center()

        # Initialize combo boxes and text labels
        text_label = Dict()
        self.__combox = Dict()
        for i in ["name", "train.id"]:
            text_label[i] = QtWidgets.QLabel()
            self.__combox[i] = QtWidgets.QComboBox()
            self.__combox[i].setFixedHeight(27)
        text_label["name"].setText("User")
        text_label["train.id"].setText("Experiment ID")

        # Layout
        sub_layout = Dict()
        layout_container = Dict()
        for i in ["name", "train.id"]:
            sub_layout[i] = QtWidgets.QVBoxLayout()
            sub_layout[i].addWidget(text_label[i])
            sub_layout[i].addWidget(self.__combox[i])
            sub_layout[i].setSpacing(1)
            layout_container[i] = QtWidgets.QWidget()
            layout_container[i].setLayout(sub_layout[i])

        # Central widget
        v_layout = QtWidgets.QVBoxLayout()
        for i in ["name", "train.id"]:
            v_layout.addWidget(layout_container[i])
        v_layout.setSpacing(30)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(v_layout)
        self.setCentralWidget(central_widget)

        # Add items to the combo boxes
        self.__combox["name"].addItems(
            [
                "~ Administrator",
                "Hanna Rahbek Mortensen",
                "Kenneth Jensen",
                "Jesper Grau Eriksen",
            ]
        )
        self.__sort_combox(self.__combox["name"])
        self.__combox["name"].setCurrentIndex(-1)

        # Connect the combo boxes to the function
        self.__combox["name"].activated.connect(self.__fill_combox_results)
        self.__combox["train.id"].activated.connect(self.__open_main_window)

    def __simplify_user_name(self):
        user_name = self.__combox["name"].currentText()
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
                debug_mode=1,
            )

        # install the event filter on the QApplication instance
        # This ensures that key press events will always trigger the main window's event handler,
        # regardless of which widget currently has focus.
        app = QtWidgets.QApplication.instance()
        app.installEventFilter(self.__main_window)


# clear cache
if 0:
    GPU.clear_cache()
    Debug.clear_debug_data()
    Debug.clear_linux_trash()


# show login window
app = QApplication(sys.argv)
app.setPalette(DarkPalette())  # dark theme
login_window = LoginWindow()
login_window.show()
sys.exit(app.exec_())
