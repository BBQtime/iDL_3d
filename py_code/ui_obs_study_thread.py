from custom import GPU
from numpy import ndarray
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt


class ObsStudyThread(QThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

    def __init__(
        self,
        progress_bar: QtWidgets.QProgressBar,
        progress_bar_label: QtWidgets.QLabel,
    ):
        super().__init__()
        self.is_running = False
        self._progress_bar = progress_bar
        self._progress_bar_label = progress_bar_label

    def stop(self):
        if self.is_running:
            self.terminate()  # force stop
            GPU.clear_cache()
            # self.quit()  # signal the thread to exit its event loop
            # self.wait()  # wait for the thread to be cleaned up properly
            self.is_running = False
            self._hide_progress_widgets()

    def _show_progress_widgets(self):
        self._progress_bar_label.show()
        self._progress_bar.setValue(0)
        self._progress_bar.show()

    def _hide_progress_widgets(self):
        self._progress_bar_label.hide()
        self._progress_bar.hide()
        self._progress_bar.setValue(0)


class ObsStudyGTVnThread(ObsStudyThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

    def set_param(
        self,
        idl_gtvn_id: str,
        dataset_ver: str,
        patient: str,
        idl_gtvn_clicks: ndarray,
        debug_mode: bool,
    ):
        self.__idl_gtvn_id = idl_gtvn_id
        self.__dataset_ver = dataset_ver
        self.__patient = patient
        self.__idl_gtvn_clicks = idl_gtvn_clicks
        self.__debug_mode = debug_mode

    def run(self):
        self._show_progress_widgets()
        self.is_running = True
        training_idl_gtvn = TrainingIDLGTVn(self.progress_signal)
        training_idl_gtvn.obs_study(
            idl_gtvn_id=self.__idl_gtvn_id,
            dataset_ver=self.__dataset_ver,
            patient=self.__patient,
            idl_gtvn_clicks=self.__idl_gtvn_clicks,
            debug_mode=self.__debug_mode,
        )
        self._hide_progress_widgets()
        self.is_running = False
        self.complete_signal.emit()


class ObsStudyGTVtThread(ObsStudyThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()

    def set_param(
        self,
        idl_gtvt_id: str,
        dataset_ver: str,
        patient: str,
        debug_mode: bool,
    ):
        self.__idl_gtvt_id = idl_gtvt_id
        self.__dataset_ver = dataset_ver
        self.__patient = patient
        self.__debug_mode = debug_mode

    def run(self):
        self._show_progress_widgets()
        self.is_running = True
        training_idl_gtvt = TrainingIDLGTVt(self.progress_signal)
        training_idl_gtvt.obs_study(
            idl_gtvt_id=self.__idl_gtvt_id,
            dataset_ver=self.__dataset_ver,
            patient=self.__patient,
            debug_mode=self.__debug_mode,
        )
        self._hide_progress_widgets()
        self.is_running = False
        self.complete_signal.emit()
