import global_utils.global_core as g
from global_utils.custom_list import List

# from research_utils import gtvt_input_slices, idl_time, idl_vs_correction, iov
from research_utils import idl_vs_correction
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

    idl_vs_correction.plot_metrics([JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID])
    idl_vs_correction.plot_metrics([JESPER_GTVN_ID, KENNETH_GTVN_ID, HANNA_GTVN_ID])

    # for obs_study_id in [
    #     JESPER_GTVT_ID,
    #     KENNETH_GTVT_ID,
    #     HANNA_GTVT_ID,
    # ]:
    #     gtvt_input_slices.calculate_comparison_metrics(obs_study_id)

    # for obs_study_id in [
    #     JESPER_GTVT_ID,
    #     KENNETH_GTVT_ID,
    #     HANNA_GTVT_ID,
    # ]:
    #     gtvt_input_slices.calculate_input_inconsistency(obs_study_id)

    # gtvt_input_slices.plot_metrics_comparison(
    #     [JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID]
    # )

    # for pair in List(
    #     [
    #         JESPER_GTVT_ID,
    #         KENNETH_GTVT_ID,
    #         HANNA_GTVT_ID,
    #         "label",
    #     ]
    # ).get_combinations(2):
    #     iov.calculate_iov(pair[0], pair[1])

    # for pair in List(
    #     [
    #         JESPER_GTVN_ID,
    #         KENNETH_GTVN_ID,
    #         HANNA_GTVN_ID,
    #         "label",
    #     ]
    # ).get_combinations(2):
    #     iov.calculate_iov(pair[0], pair[1])

    # iov.plot_iov()

    # idl_time.plot_time_per_patient(
    #     [JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID]
    # )

    # idl_time.plot_time_per_step(
    #     [JESPER_GTVT_ID, KENNETH_GTVT_ID, HANNA_GTVT_ID]
    # )

    print("Done!")
