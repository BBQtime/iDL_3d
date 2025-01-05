class ErrMsg:
    DATASET_VER_INVALID = "Invalid value for dataset_ver!"
    DATASET_PART_INVALID = "Invalid value for dataset_part!"
    OBS_STUDY_STEP_INVALID = "Invalid value for obs study step!"


class SelectScenario:
    LARGEST = "largest"
    GRAVITY_CENTER = "gravity.center"
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


class ObsStudyGTVtStep:
    CLICK_CENTER = "click.center"
    DELINEATE = "delineate"
    WAIT_PRED = "wait.pred"
    CORRECT = "correct"
    APPROVED = "approved"


class ObsStudyGTVnStep:
    CLICK_CENTERS = "click.centers"
    WAIT_PRED = "wait.pred"
    CORRECT = "correct"
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
