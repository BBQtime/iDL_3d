import global_utils.global_core as g
from global_utils.str_lib import DatasetPart, DatasetVer, MdaObs
from research_utils.research_core import calculate_idl_gtvs_metric

if __name__ == "__main__":

    g.clear_gpu_cache()
    g.clear_linux_trash()
    g.clear_debug_data()

    # calculate_idl_gtvs_metric(
    #     idl_gtvt_id="idl.gtvt_2023.07.21.01.40.28",
    #     idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
    # )

    # for obs_study_id in [
    #     ObsStudyID.JESPER_GTVT,
    #     ObsStudyID.KENNETH_GTVT,
    #     ObsStudyID.HANNA_GTVT,
    #     ObsStudyID.JESPER_GTVN,
    #     ObsStudyID.KENNETH_GTVN,
    #     ObsStudyID.HANNA_GTVN,
    # ]:
    #     calculate_3d_idl_vs_correct(obs_study_id)

    # plot_3d_idl_vs_correct(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )
    # plot_3d_idl_vs_correct(
    #     [ObsStudyID.JESPER_GTVN, ObsStudyID.KENNETH_GTVN, ObsStudyID.HANNA_GTVN]
    # )

    # for obs_study_id in [
    #     ObsStudyID.JESPER_GTVT,
    #     ObsStudyID.KENNETH_GTVT,
    #     ObsStudyID.HANNA_GTVT,
    # ]:
    #     calculate_gtvt_slices_metrics(obs_study_id)

    # for obs_study_id in [
    #     ObsStudyID.JESPER_GTVT,
    #     ObsStudyID.KENNETH_GTVT,
    #     ObsStudyID.HANNA_GTVT,
    # ]:
    #     calculate_gtvt_input_variation(obs_study_id)

    # plot_gtvt_slices_metrics(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )

    # for pair in List(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT, "label"]
    # ).get_combinations(2):
    #     calculate_iov(pair[0], pair[1])

    # for pair in List(
    #     [ObsStudyID.JESPER_GTVN, ObsStudyID.KENNETH_GTVN, ObsStudyID.HANNA_GTVN, "label"]
    # ).get_combinations(2):
    #     calculate_iov(pair[0], pair[1])

    # plot_iov()

    # plot_time_per_patient(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )

    # plot_time_per_step(
    #     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
    # )

    print("Done!")
