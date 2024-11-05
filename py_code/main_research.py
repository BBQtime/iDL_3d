import global_utils.global_core as g
from global_utils.custom_list import List
from global_utils.str_lib import ObsStudyID
from research_utils import gtvt_input_slices, idl_time, idl_vs_correction, iov
from research_utils.research_core import calculate_idl_gtvs_metric, update_font_size

if __name__ == "__main__":
    g.clear_gpu_cache()
    g.clear_linux_trash()
    g.clear_debug_data()

    update_font_size()

    calculate_idl_gtvs_metric(
        idl_gtvt_id="idl.gtvt_au.ext_center",
        idl_gtvn_id="idl.gtvn_au_multi.clicks",
    )

    # for obs_study_id in [
    #     ObsStudyID.JESPER_GTVT,
    #     ObsStudyID.KENNETH_GTVT,
    #     ObsStudyID.HANNA_GTVT,
    #     ObsStudyID.JESPER_GTVN,
    #     ObsStudyID.KENNETH_GTVN,
    #     ObsStudyID.HANNA_GTVN,
    # ]:
    #     idl_vs_correction.calculate_metrics(obs_study_id)

    # idl_vs_correction.plot_metrics(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )
    # idl_vs_correction.plot_metrics(
    #     [ObsStudyID.JESPER_GTVN, ObsStudyID.KENNETH_GTVN, ObsStudyID.HANNA_GTVN]
    # )

    # for obs_study_id in [
    #     ObsStudyID.JESPER_GTVT,
    #     ObsStudyID.KENNETH_GTVT,
    #     ObsStudyID.HANNA_GTVT,
    # ]:
    #     gtvt_input_slices.calculate_comparison_metrics(obs_study_id)

    # for obs_study_id in [
    #     ObsStudyID.JESPER_GTVT,
    #     ObsStudyID.KENNETH_GTVT,
    #     ObsStudyID.HANNA_GTVT,
    # ]:
    #     gtvt_input_slices.calculate_input_inconsistency(obs_study_id)

    # gtvt_input_slices.plot_metrics_comparison(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )

    # for pair in List(
    #     [
    #         ObsStudyID.JESPER_GTVT,
    #         ObsStudyID.KENNETH_GTVT,
    #         ObsStudyID.HANNA_GTVT,
    #         "label",
    #     ]
    # ).get_combinations(2):
    #     iov.calculate_iov(pair[0], pair[1])

    # for pair in List(
    #     [
    #         ObsStudyID.JESPER_GTVN,
    #         ObsStudyID.KENNETH_GTVN,
    #         ObsStudyID.HANNA_GTVN,
    #         "label",
    #     ]
    # ).get_combinations(2):
    #     iov.calculate_iov(pair[0], pair[1])

    # iov.plot_iov()

    # idl_time.plot_time_per_patient(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )

    # idl_time.plot_time_per_step(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )

    print("Done!")
