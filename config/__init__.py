DATA_BASE_PATH = "/home/brainlab/Dataset/ISRUC-Sleep/Group1"

EEG = [
    # EEG
    # CASE 1
    # 'F3-M2', 'C3-M2', 'O1-M2',
    'F4-M1', 'C4-M1', 'O2-M1',
    # CASE 2
    # 'F3-A2', 'C3-A2', 'O1-A2',
    'F4-A1', 'C4-A1', 'O2-A1',

    # ECG
    # 'X2'
]

CHANNEL_LENGTH = 3
# CHANNEL_LENGTH = 4

SLEEP_STAGE_ENCODING = {
    # awake
    'W': 0,

    # sleep
    # 'N': 1,
    'N': 1,
    'N1': 1,
    'N2': 2,
    'N3': 3,

    # REM
    'R': 4
}

'''
AWAKE: Awakening
CH: Central Hypopnea
CA: Central Apnea
OH: Obstructive Hypopnea
OA: Obstructive Apnea
AR: Arousal
MH: Mixed Hypopnea
'''

EVENT_ENCODING = {
    'OA': 0,
    'CA': 0,

    # 'AWAKE': 0,
    # 'OH': 1,
    # 'CH': 1,
    # 'MH': 1,
    # 'OA': 2,
    # 'CA': 2,
    # 'AR': 3,

    # 'AWAKE': 0,
    # 'OH': 1,
    # 'OA': 2,
    # 'AR': 3,

    # 'MCHG': 0,
    # 'AWAKE': 1,
    # 'CH': 2,
    # 'CA': 3,
    # 'OH': 4,
    # 'OA': 5,
    # 'REM': 6,
    # 'AR': 7,
    # 'MH': 8,
}

EVENT_ENCODING_VALUES = set(EVENT_ENCODING.values())
SLEEP_STAGE_ENCODING_VALUES = set(SLEEP_STAGE_ENCODING.values())

ALL_SUBJECTS = [i for i in range(100)]
