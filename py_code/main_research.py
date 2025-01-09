import global_utils.global_core as g
from global_utils.custom_list import List
from research_utils import gtvt_input_slices, idl_time, idl_vs_correction, iov
from research_utils.research_core import (
    HANNA_GTVN_ID,
    HANNA_GTVT_ID,
    JESPER_GTVN_ID,
    JESPER_GTVT_ID,
    KENNETH_GTVN_ID,
    KENNETH_GTVT_ID,
    calculate_idl_gtvs_metric,
)

if __name__ == "__main__":
    g.clear_gpu_cache()
    g.clear_linux_trash()
    g.clear_debug_data()

    gtvt_input_slices.plot_bias_gtvt_center()

    # calculate_idl_gtvs_metric(
    #     idl_gtvt_id="idl.gtvt_au.ext_center",
    #     idl_gtvn_id="idl.gtvn_au_multi.clicks",
    # )

    # for obs_study_id in [
    #     JESPER_GTVT_ID,
    #     KENNETH_GTVT_ID,
    #     HANNA_GTVT_ID,
    #     JESPER_GTVN_ID,
    #     KENNETH_GTVN_ID,
    #     HANNA_GTVN_ID,
    # ]:
    #     idl_vs_correction.calculate_metrics(obs_study_id)

    # obs_study_id_list = [
    #     JESPER_GTVT_ID,
    #     KENNETH_GTVT_ID,
    #     HANNA_GTVT_ID,
    #     JESPER_GTVN_ID,
    #     KENNETH_GTVN_ID,
    #     HANNA_GTVN_ID,
    # ]
    # idl_vs_correction.create_metrics_tables(obs_study_id_list)

    # idl_vs_correction.plot_metrics_no_apl(
    #     [JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID]
    # )
    # idl_vs_correction.plot_metrics_no_apl(
    #     [JESPER_GTVN_ID, KENNETH_GTVN_ID, HANNA_GTVN_ID]
    # )

    # idl_vs_correction.plot_metrics_apl(
    #     obs_study_gtvt_id_list=[JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID],
    #     obs_study_gtvn_id_list=[JESPER_GTVN_ID, KENNETH_GTVN_ID, HANNA_GTVN_ID],
    # )

    # for obs_study_id in [
    #     JESPER_GTVT_ID,
    #     KENNETH_GTVT_ID,
    #     HANNA_GTVT_ID,
    # ]:
    #     gtvt_input_slices.calculate_metrics(obs_study_id)

    # gtvt_input_slices.create_metrics_tables(
    #     [JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID]
    # )

    # for gtv_list in [
    #     [JESPER_GTVN_ID, KENNETH_GTVN_ID, HANNA_GTVN_ID, "label"],
    #     [JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID, "label"],
    # ]:
    #     gtv_list = List(gtv_list)
    #     for pair in gtv_list.get_combinations(2):
    #         iov.calculate_iov(pair[0], pair[1])

    # iov.create_median_table()

    # iov.plot_iov()

    # idl_time.plot_time_per_patient([JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID])

    # idl_time.plot_time_per_step([JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID])

    print("Done!")
