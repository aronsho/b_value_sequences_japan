# python 0_parameters_main.py

# ========= IMPORTS =========
from pathlib import Path

import numpy as np
import pandas as pd

# ========= CONFIG =========
DIR = Path("data")
SHAPE_DIR = Path("data/shape_japan")
CAT_DIR = Path("data/catalogs")

# ========= PARAMETERS ======
DIMENSION = 3                 # 2 or 3
BUFFER_M = 40_000            # 30 km

# transformation info
EPSG_GEOGRAPHIC = 4326      # WGS84
EPSG_JAPAN_M = 6677         # Japan (meters)
p1 = np.array([35.018970, 135.769601, 0])  # reference point 1
p2 = np.array([36.018970, 135.769601, 0])  # reference point 2

# general parameters for catalog / b-value estimation
MC_FIXED = 0.7              # fixed magnitude of completeness
DELTA_M = 0.1               # magnitude binning width
CORRECTION_FACTOR = 0.2     # correction factor for Mc (maxc)
DMC = 0.1                   # mc for differences
MIN_N_M = 50                # minimum number of mags for b-value estimate
B_METHOD = 'global'         # method for b-value estimation

# parameters for sequences (main results)
MAGNITUDE_THRESHOLD = 6.0       # mainshock magnitude threshold
DEPTH_THRESHOLD = 150           # depth threshold for catalog
TIME_START = "2000-01-01"        # catalog start time
RUPTURE_RELATION = "surface"    # rupture length relation type
DAYS_AFTER = 100                # time window after main event
DAYS_BEFORE = 10 * 365          # time window before main event
DIMENSION = 3                   # 2D or 3D
RADIUS_FAR = 2.0                # radius multiplier for "far"
RADIUS_CLOSE = 0.5              # radius multiplier for "close"
EXCLUDE_AFTERSHOCK_DAYS = 1     # number of aftershocks excluded from analysis
MIN_N_SEQ = 200                 # minimum number per seq to take into account

# for map
N_FULL_MAP = 25600            # number of tiles for full map

# evaluation
P_THRESHOLD = 0.05           # threshold for p-value

# ========= SAVE ============# Explicit dictionary of parameters and config
variables_dict = {
    "SHAPE_DIR": SHAPE_DIR,
    "CAT_DIR": CAT_DIR,
    "DIMENSION": DIMENSION,
    "BUFFER_M": BUFFER_M,
    "EPSG_GEOGRAPHIC": EPSG_GEOGRAPHIC,
    "EPSG_JAPAN_M": EPSG_JAPAN_M,
    "p1": p1.tolist(),
    "p2": p2.tolist(),
    "MC_FIXED": MC_FIXED,
    "DELTA_M": DELTA_M,
    "CORRECTION_FACTOR": CORRECTION_FACTOR,
    "DMC": DMC,
    "MIN_N_M": MIN_N_M,
    "B_METHOD": B_METHOD,
    "MAGNITUDE_THRESHOLD": MAGNITUDE_THRESHOLD,
    "DEPTH_THRESHOLD": DEPTH_THRESHOLD,
    "TIME_START": TIME_START,
    "RUPTURE_RELATION": RUPTURE_RELATION,
    "DAYS_AFTER": DAYS_AFTER,
    "DAYS_BEFORE": DAYS_BEFORE,
    "RADIUS_FAR": RADIUS_FAR,
    "RADIUS_CLOSE": RADIUS_CLOSE,
    "EXCLUDE_AFTERSHOCK_DAYS": EXCLUDE_AFTERSHOCK_DAYS,
    "MIN_N_SEQ": MIN_N_SEQ,
    "N_FULL_MAP": N_FULL_MAP,
    "P_THRESHOLD": P_THRESHOLD,
}

# Save as wide CSV (one row, many columns)
pd.DataFrame([variables_dict]).to_csv(DIR / "variables.csv", index=False)
