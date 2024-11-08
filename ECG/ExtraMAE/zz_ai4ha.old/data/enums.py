from enum import Enum


class TrainingSplitModality(Enum):
    TRAIN_ONLY = 0
    TEST_ONLY = 1
    TRAIN_VAL = 2
    TRAIN_TEST = 3
    TRAIN_VAL_TEST = 4


class DatasetID(Enum):
    SUN_SEG = 'sun-seg'
    KVASIR_SEG = 'kvasir-seg'
    LD_POLYP = 'ld-polyp'
    LD_POLYP_GEN = 'ld-polyp-gen'