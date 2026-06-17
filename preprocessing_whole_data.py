import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
INPUT_CSV = r"C:/CARLA_0.9.13/NGSIM/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260221_I-80.csv"
OUTPUT_CSV = r"C:/CARLA_0.9.13/NGSIM/I80_all_clean_mainline.csv"

MIN_DURATION_MS = 8000          # keep vehicles appearing >= 8 sec
TARGET_HZ = 5                   # NGSIM nominally 10 Hz -> keep every 2nd frame
ORIGINAL_HZ = 10
DO_DOWNSAMPLE = True

DO_SMOOTHING = True
SMOOTH_WINDOW = 3               # 3 frames after downsampling = 1 sec at 5 Hz

KEEP_ONLY_MAIN_LANES = True
MAIN_LANES = {1, 2, 3, 4, 5}

# ==============================
# READ
# ==============================
df = pd.read_csv(INPUT_CSV, low_memory=False)

needed_cols = [
    "Vehicle_ID", "Frame_ID", "Global_Time",
    "Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID"
]
df = df[needed_cols].copy()

# numeric conversion
num_cols = ["Frame_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "Lane_ID"]).reset_index(drop=True)

# sort first
df = df.sort_values(by=["Vehicle_ID", "Global_Time"]).reset_index(drop=True)

print("Rows after basic cleaning:", len(df))
print("Unique vehicles:", df["Vehicle_ID"].nunique())

# ==============================
# OPTIONAL: keep only mainline lanes
# ==============================
if KEEP_ONLY_MAIN_LANES:
    df = df[df["Lane_ID"].isin(MAIN_LANES)].reset_index(drop=True)
    print("Rows after keeping only main lanes 1-5:", len(df))

# ==============================
# REMOVE SHORT TRACKS
# ==============================
dur = df.groupby("Vehicle_ID")["Global_Time"].agg(["min", "max"])
dur["duration"] = dur["max"] - dur["min"]

valid_vehicles = dur[dur["duration"] >= MIN_DURATION_MS].index
df = df[df["Vehicle_ID"].isin(valid_vehicles)].reset_index(drop=True)

print("Rows after removing short tracks:", len(df))
print("Vehicles after removing short tracks:", df["Vehicle_ID"].nunique())

# ==============================
# DOWNSAMPLE EARLY
# ==============================
# Since NGSIM is nominally 10 Hz, keep every 2nd row per vehicle to get ~5 Hz
# We do this BEFORE smoothing and before any sequence building.
# ==============================
if DO_DOWNSAMPLE and TARGET_HZ < ORIGINAL_HZ:
    step = ORIGINAL_HZ // TARGET_HZ
    df = (
        df.groupby("Vehicle_ID", group_keys=False)
          .apply(lambda g: g.iloc[::step])
          .reset_index(drop=True)
    )
    print(f"Rows after downsampling to ~{TARGET_HZ} Hz:", len(df))

# ==============================
# OPTIONAL SMOOTHING
# ==============================
def smooth_vehicle(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["Local_X"] = g["Local_X"].rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
    g["Local_Y"] = g["Local_Y"].rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
    g["v_Vel"]   = g["v_Vel"].rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
    return g

if DO_SMOOTHING:
    df = df.groupby("Vehicle_ID", group_keys=False).apply(smooth_vehicle)
    df = df.dropna(subset=["Local_X", "Local_Y", "v_Vel"]).reset_index(drop=True)
    print("Rows after smoothing:", len(df))

# final sort
df = df.sort_values(by=["Vehicle_ID", "Global_Time"]).reset_index(drop=True)

# save
df.to_csv(OUTPUT_CSV, index=False)
print("Saved clean file to:", OUTPUT_CSV)
