import os
from datetime import datetime, timedelta

import global_utils.global_core as g


class ObsStudyTimer:
    # key names for dict
    CLICK_GTVT_CENTER = "click.gtvt.center"
    CLICK_GTVN_CENTERS = "click.gtvn.centers"
    DELINEATE_GTVT = "delineate.gtvt"
    WAIT_GTVT_PRED = "wait.gtvt.pred"
    WAIT_GTVN_PRED = "wait.gtvn.pred"
    CORRECT_GTVT = "correct.gtvt"
    CORRECT_GTVN = "correct.gtvn"

    def __init__(
        self,
        baseline_id: str,
        idl_gtvt_id: str,
        patient: str,
        timer_name: str,
    ):
        self.__start_time = None
        self.__patient = patient
        self.__timer_name = timer_name
        # json path
        self.__json_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            baseline_id,
            idl_gtvt_id,
            "time_used.json",
        )
        # create an empty json file
        if not os.path.exists(self.__json_path):
            g.save_json({}, self.__json_path)

    def start(self):
        self.__start_time = datetime.now()

    def end(self):
        if self.__start_time is None:
            return

        duration = datetime.now() - self.__start_time
        self.__start_time = None
        total_seconds = int(duration.total_seconds())
        # create a new timedelta without microseconds
        duration = timedelta(seconds=total_seconds)
        duration = str(duration)
        # save json
        time_log = g.load_json(self.__json_path)
        time_log["patient={}".format(self.__patient)][self.__timer_name] = duration
        g.save_json(time_log, self.__json_path)
