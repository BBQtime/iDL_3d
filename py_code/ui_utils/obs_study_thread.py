# from training_utils.idl_gtvt_training import IDLGTVtTraining
import time

import global_utils.global_core as g
from numpy import ndarray
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from training_utils.idl_gtvn_training import IDLGTVnTraining


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
            g.clear_gpu_cache()
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
        idl_gtvn_training = IDLGTVnTraining(self.progress_signal)
        idl_gtvn_training.obs_study(
            idl_gtvn_id=self.__idl_gtvn_id,
            dataset_ver=self.__dataset_ver,
            patient=self.__patient,
            obs_gtvn_clicks=self.__idl_gtvn_clicks,
            debug_mode=self.__debug_mode,
            device_id=0,
        )
        self._hide_progress_widgets()
        self.is_running = False
        self.complete_signal.emit()


class ObsStudyGTVtProgressThread(ObsStudyThread):
    progress_signal = pyqtSignal(float)
    complete_signal = pyqtSignal()
    queue = None
    # def set_param(
    #     self,
    #     idl_gtvt_id: str,
    #     dataset_ver: str,
    #     patient: str,
    #     debug_mode: bool,
    # ):
    #     self.__idl_gtvt_id = idl_gtvt_id
    #     self.__dataset_ver = dataset_ver
    #     self.__patient = patient
    #     self.__debug_mode = debug_mode

    def run(self):
        self.is_running = True
        self._show_progress_widgets()
        try:
            while self.is_running:
                # Check the queue for messages from the training process
                if not self.queue.empty():
                    message = self.queue.get()
                    if "progress" in message:
                        progress_value = message["progress"]
                        self.progress_signal.emit(progress_value)

                    if "status" in message and message["status"] == "complete":
                        self.complete_signal.emit()
                        self.is_running = False
                        break

                # Sleep for a short time to prevent overloading the CPU
                time.sleep(0.5)

        except Exception as e:
            print(f"Error in thread: {str(e)}")
        finally:
            self.is_running = False
            self._hide_progress_widgets()
            self.complete_signal.emit()

    def stop(self):
        self.is_running = False
        self._hide_progress_widgets()


# class ObsStudyGTVtThread(ObsStudyThread):
#     progress_signal = pyqtSignal(float)
#     complete_signal = pyqtSignal()

#     def set_param(
#         self,
#         idl_gtvt_id: str,
#         dataset_ver: str,
#         patient: str,
#         debug_mode: bool,
#     ):
#         self.__idl_gtvt_id = idl_gtvt_id
#         self.__dataset_ver = dataset_ver
#         self.__patient = patient
#         self.__debug_mode = debug_mode

#     def run(self):
#         self._show_progress_widgets()
#         self.is_running = True
#         # training_idl_gtvt = TrainingIDLGTVt(self.progress_signal)
#         # training_idl_gtvt.obs_study(
#         #     idl_gtvt_id=self.__idl_gtvt_id,
#         #     dataset_ver=self.__dataset_ver,
#         #     patient=self.__patient,
#         #     debug_mode=self.__debug_mode,
#         # )
#         self._hide_progress_widgets()
#         self.is_running = False
#         self.complete_signal.emit()
