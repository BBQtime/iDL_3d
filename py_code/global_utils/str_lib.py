class ErrMsg:
    DATASET_VER_INVALID = "Value of dataset_ver is invalid!"
    DATASET_PART_INVALID = "Value of dataset_part is invalid!"


class SelectScenario:
    LARGEST = "largest"
    GRAVITY_CENTER = "gravity.center"
    BIAS_GRAVITY_CENTER = "bias.gravity.center"
    EQUAL_DIVIDE = "equal.divide"
    RANDOM = "random"
    USER_CLICK = "user.click"


class DrawingMode:
    GTVT_PEN = "gtvt.pen"
    GTVN_PEN = "gtvn.pen"
    GTVT_ERASER = "gtvt.eraser"
    GTVN_ERASER = "gtvn.eraser"
    GTVT_CLEAR = "gtvt.clear"
    GTVN_CLEAR = "gtvn.clear"
    GTVT_RESTORE = "gtvt.restore"
    GTVN_RESTORE = "gtvn.restore"


class ObsStudyID:
    JESPER_GTVT = "idl.gtvt_2024.03.18.09.05.54_Jesper_research"
    KENNETH_GTVT = "idl.gtvt_2024.04.12.12.05.44_Kenneth_research"
    HANNA_GTVT = "idl.gtvt_2024.04.18.11.04.48_Hanna_research"
    JESPER_GTVN = "idl.gtvn_2024.03.18.09.05.54_Jesper_research"
    KENNETH_GTVN = "idl.gtvn_2024.04.12.12.05.44_Kenneth_research"
    HANNA_GTVN = "idl.gtvn_2024.04.18.11.04.48_Hanna_research"


class ObsStudyStep:
    SELECT_PATIENT = "select.patient"

    CLICK_GTVT_CENTER = "click.gtvt.center"

    DRAW_GTVT = "draw.gtvt"  # for "obs study step" and  "step label"
    DRAW_GTVT_TRANSVERSE = "draw.gtvt.transverse"  # only for "step label"
    DRAW_GTVT_CORONAL = "draw.gtvt.coronal"  # only for "step label"
    DRAW_GTVT_SAGITTAL = "draw.gtvt.sagittal"  # only for "step label"

    CLICK_GTVN_CENTER = "click.gtvn.center"

    WAITING = "waiting"  # for "obs study step" and "step label"
    WAITING_GTVT = "waiting.gtvt"  # only for "timer"
    WAITING_GTVN = "waiting.gtvn"  # only for "timer"

    CORRECT_GTVT = "correct.gtvt"  # "obs study step", "step label" and "timer"
    CORRECT_GTVN = "correct.gtvn"  # "obs study step", "step label" and "timer"
    CORRECT_BOTH = "correct.both"  # only for "obs study step"

    APPROVED = "approved"


class DatasetVer:
    AU = "au"
    AU_EXT = "au.ext"
    OBS_STUDY = "obs.study"
    MDA = "mda"
    NKI = "nki"
    HECKTOR = "hecktor"


class MdaObs:
    AAA = "AAA"
    DMEl = "DMEl"
    MRA = "MRA"
    SA = "SA"
    YK = "YK"


class DatasetPart:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DisplayMode:
    MODAL_FIXED = "modal.fixed"
    PLANE_FIXED = "plane.fixed"


class Modal:
    CT = "ct"
    PT = "pt"
    MR1 = "mr1"
    MR2 = "mr2"


class Metric:
    DSC = "dsc"
    MSD = "msd"
    HD95 = "hd95"
    APL_VOXEL = "apl.voxel"
    APL_PCT = "apl.pct"
    SDSC = "sdsc"


class Plane:
    TRANSVERSE = "transverse"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


class Stat:
    MEDIAN = "median"
    AVG = "avg"
