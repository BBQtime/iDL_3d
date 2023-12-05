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


class IDLStep:
    CLICK_GTVT_CENTER = "click.gtvt.center"
    DRAW_GTVT = "draw.gtvt"
    CLICK_GTVN_CENTER = "click.gtvn.center"
    CORRECTION = "correction"


class DatasetPart:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    TEST_INTER = "test.inter"
    TEST_EXTER = "test.exter"


# ui-display mode
DISPLAY_MODE = "display.mode"


class DisplayMode:
    MODAL_FIXED = "modal.fixed"
    PLANE_FIXED = "plane.fixed"


# dataset versions
AU_3MM = "au.3mm"
AU_1MM = "au.1mm"
MDA = "mda"


# ui
MODAL = "modal"
PLANE = "plane"

# ui color enhancement
COLOR_ENHANCE = "color.enhance"
BRIGHT = "bright"
CONTRAST = "contrast"

ZOOM = "zoom"
CT_PT_MIX = "ct.pt.mix"

# annotation tools
ANNOTATION = "annotation"
PEN = "pen"
PEN_SIZE = "pen.size"
DRAW_GTVT = "draw.gtvt"
DRAW_GTVN = "draw.gtvn"
ERASER = "eraser"
CLEAR = "clear"
CONFIRM = "confirm"


# anatomical planes
TRANSVERSE = "transverse"
CORONAL = "coronal"
SAGITTAL = "sagittal"

# ui 3d img names
CLICK = "click"
CLICKS = "clicks"
GTVT_CLICK = "gtvt.click"
GTVN_CLICKS = "gtvn.clicks"

GTVT_ANNOTATION = "gtvt.annotation"

GTVT_CORRECTION = "gtvt.correction"
GTVT_CORRECTION_MASK = "gtvt.correction.mask"
GTVN_CORRECTION = "gtvn.correction"
GTVN_CORRECTION_MASK = "gtvn.correction.mask"
CORRECTION = "correction"
CORRECTION_MASK = "correction.mask"

GTVT_LABEL = "gtvt.label"
GTVN_LABEL = "gtvn.label"
GTVT_PRED = "gtvt.pred"
GTVN_PRED = "gtvn.pred"
GTVT_PRED_FINAL = "gtvt.pred.final"
GTVN_PRED_FINAL = "gtvn.pred.final"


# training type
BASELINE = "baseline"
IDL_GTVT = "idl.gtvt"
IDL_GTVN = "idl.gtvn"

# patients
PATIENT = "patient"

# modalities
CT = "ct"
PT = "pt"
MR1 = "mr1"
MR2 = "mr2"

# segmentation metrics
DSC = "dsc"
MSD = "msd"
HD95 = "hd95"


# gtv
GTVT = "gtvt"
GTVN = "gtvn"
GTVS = "gtvs"

# channels
BACKGROUND = "background"
SEED = "seed"
LABEL = "label"
PRED = "pred"
DISTANCE_MAP = "distance.map"

# scores
MEDIAN = "median"
AVG = "avg"

# augmentation methods
ELASTIC = "elastic"
SCALE = "scale"
TRANSLATE = "translate"
ROTATE = "rotate"
FLIP_LR = "flip.lr"
FLIP_UD = "flip.ud"

# hyperparameters
DEVICE = "device"
NO_PT = "no.pt"
DATASET_VER = "dataset.ver"

OPTIM = "optim"
ADAM = "adam"
SCHEDULER = "scheduler"
REDUCE_LR_ON_PLATEAU = "reduce.lr.on.plateau"
DROPOUT = "dropout"
LAYER_FREEZING = "layer.freezing"

UNIFIED_FOCAL_LOSS = "unified.focal.loss"
LOSS_FUNC = "loss.func"
LOSS_ASYM = "loss.asym"
LOSS_WEIGHT = "loss.weight"
LOSS_DELTA = "loss.delta"
LOSS_GAMMA = "loss.gamma"

AUGMENT_METHODS = "augment.methods"
AUGMENT_PCT = "augment.pct"
AUGMENT_TIMES = "augment.times"
AUGMENT_MIN = "augment.min"
AUGMENT_MAX = "augment.max"

EPOCHS = "epochs"
EPOCHS_ACTUAL = "epochs.actual"
EARLY_STOP_EPOCHS = "early.stop.epochs"
KEEP_BEST_CNN_NUM = "keep.best.cnn.num"
ITER = "iter"

LR = "lr"
LR_ACTUAL = "lr.actual"
LR_MIN = "lr.min"
LR_DECAY_FACTOR = "lr.decay.factor"
LR_DECAY_PATIENCE = "lr.decay.patience"

BATCH_SIZE = "batch.size"
BATCH_SIZE_ACTUAL = "batch.size.actual"

SELECT_SCENARIO = "select.scenario"
SELECT_STEP_TRANSVERSE = "select.step.transverse"
SELECT_STEP_CORONAL = "select.step.coronal"
SELECT_STEP_SAAGITTAL = "select.step.sagittal"

WEIGHT_BACKGROUND = "weight.background"
WEIGHT_DISTANCE_STEP = "weight.distance.step"
WEIGHT_FP_FN = "weight.fp.fn"
WEIGHT_PREV_ROUND_DECAY = "weight.prev.round.decay"
WEIGHT_SLICE = "weight.slice"
WEIGHT_MAP = "weight.map"

TIME_SPENT = "time.spent"
TIME_SPENT_TOTAL = "time.spent.total"

TRAIN_LOADER = "train.loader"
VALID_LOADER = "valid.loader"

# CNN
UNET_PP_SLIM = "unet.pp.slim"
UNET_SLIM = "unet.slim"
CNN = "cnn"
