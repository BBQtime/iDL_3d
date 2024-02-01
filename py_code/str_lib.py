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


class IDLStep:
    SELECT_PATIENT = "select.patient"
    CLICK_GTVT_CENTER = "click.gtvt.center"
    DRAW_GTVT = "draw.gtvt"
    DRAW_GTVT_TRANSVERSE = "draw.gtvt.transverse"
    DRAW_GTVT_CORONAL = "draw.gtvt.coronal"
    DRAW_GTVT_SAGITTAL = "draw.gtvt.sagittal"
    CLICK_GTVN_CENTER = "click.gtvn.center"
    WAITING = "waiting"
    CORRECT_GTVT = "correct.gtvt"
    CORRECT_GTVN = "correct.gtvn"
    CORRECT_BOTH = "correct.both"
    APPROVED = "approved"


class DatasetPart:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    TEST_INTER = "test.inter"
    TEST_EXTER = "test.exter"


class DisplayMode:
    MODAL_FIXED = "modal.fixed"
    PLANE_FIXED = "plane.fixed"


class DatasetVer:
    AU_3MM = "au.3mm"
    AU_1MM = "au.1mm"
    MDA = "mda"


class Modal:
    CT = "ct"
    PT = "pt"
    MR1 = "mr1"
    MR2 = "mr2"


class Metric:
    DSC = "dsc"
    MSD = "msd"
    HD95 = "hd95"


class Plane:
    TRANSVERSE = "transverse"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


class Stat:
    MEDIAN = "median"
    AVG = "avg"
