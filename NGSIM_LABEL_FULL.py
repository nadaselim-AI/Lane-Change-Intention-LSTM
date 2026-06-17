import pandas as pd
import numpy as np
import time

# ==========================================
# CONFIG
# ==========================================
CSV_PATH = r"C:/CARLA_0.9.13/NGSIM/I80_all_clean_mainline.csv"
OUTPUT_NPZ = r"C:/CARLA_0.9.13/NGSIM/I80_all_samples_3s5s_fixed.npz"

HISTORY_SECONDS = 3
FUTURE_SECONDS  = 5
HZ = 5

H = HISTORY_SECONDS * HZ   # 15 frames
F = FUTURE_SECONDS  * HZ   # 25 frames

VID_COL  = "Vehicle_ID"
TIME_COL = "Global_Time"
LANE_COL = "Lane_ID"

LANE_WIDTH = 3.7
D_MAX = 100.0

MAIN_LANES = {1, 2, 3, 4, 5}
KEEP_ONLY_MAINLINE_SAMPLES = True

# ------------------------------------------
# dx convention
# We assume:
# left side  => +LANE_WIDTH
# right side => -LANE_WIDTH
# If your Local_X convention is opposite, set LEFT_POSITIVE = False
# ------------------------------------------
LEFT_POSITIVE = False

# ------------------------------------------
# labeling refinement
# only label lane change if it happens in the NEAR future
# e.g. first 5 frames at 5Hz = first 1 sec
# ------------------------------------------
NEAR_FUTURE_FRAMES = 5

# ==========================================
# LABELING
# ==========================================
def get_maneuver_label(current_lane: int, future_lanes: np.ndarray) -> int:
    """
    0 -> keep lane
    1 -> lane change left
    2 -> lane change right

    We only look at near-future frames, not the whole 5 seconds equally.
    """
    near_future = future_lanes[:NEAR_FUTURE_FRAMES]

    for lane in near_future:
        if lane != current_lane:
            if lane < current_lane:
                return 1  # LEFT
            else:
                return 2  # RIGHT
    return 0

# ==========================================
# GEOMETRIC LANE FLAGS
# ==========================================
def lane_flags_from_geometry(ego_lane: int) -> list:
    """
    CL/CR reflect lane existence, not whether another car is currently there.
    Since we keep main lanes 1..5:
      CL = 1 if there is a lane to the left of ego lane
      CR = 1 if there is a lane to the right of ego lane
    """
    CL = 1 if (ego_lane - 1) in MAIN_LANES else 0
    CR = 1 if (ego_lane + 1) in MAIN_LANES else 0
    return [CL, CR]

# ==========================================
# SURROUNDING VEHICLES
# Roles:
# F  = same lane front
# B  = same lane rear
# LF = left front
# LR = left rear
# RF = right front
# RR = right rear
# ==========================================
#btgeeb alsurroundings 7wleen alego f la7za mo3yna
def get_surroundings(df_t: pd.DataFrame, ego_row: pd.Series) -> dict:
    ego_y = float(ego_row["Local_Y"])
    ego_lane = int(ego_row[LANE_COL])

    sur = {k: None for k in ["F", "B", "LF", "LR", "RF", "RR"]}

    for _, r in df_t.iterrows():
        if r[VID_COL] == ego_row[VID_COL]:
            continue

        lane = int(r[LANE_COL])
        dy = float(r["Local_Y"] - ego_y)

        # same lane
        if lane == ego_lane:
            if dy > 0:
                # front
                if sur["F"] is None or dy < float(sur["F"]["Local_Y"] - ego_y):
                    sur["F"] = r
            elif dy < 0:
                # rear
                if sur["B"] is None or abs(dy) < abs(float(sur["B"]["Local_Y"] - ego_y)):
                    sur["B"] = r

        # left lane
        elif lane == ego_lane - 1:
            if dy > 0:
                if sur["LF"] is None or dy < float(sur["LF"]["Local_Y"] - ego_y):
                    sur["LF"] = r
            elif dy < 0:
                if sur["LR"] is None or abs(dy) < abs(float(sur["LR"]["Local_Y"] - ego_y)):
                    sur["LR"] = r

        # right lane
        elif lane == ego_lane + 1:
            if dy > 0:
                if sur["RF"] is None or dy < float(sur["RF"]["Local_Y"] - ego_y):
                    sur["RF"] = r
            elif dy < 0:
                if sur["RR"] is None or abs(dy) < abs(float(sur["RR"]["Local_Y"] - ego_y)):
                    sur["RR"] = r

    return sur

# ==========================================
# PLACEHOLDER ENCODING
# ==========================================
def side_dx(role: str) -> float:
    left_val = +LANE_WIDTH if LEFT_POSITIVE else -LANE_WIDTH
    right_val = -LANE_WIDTH if LEFT_POSITIVE else +LANE_WIDTH

    if role in ["LF", "LR"]:
        return left_val
    if role in ["RF", "RR"]:
        return right_val
    return 0.0  # F or B

def encode_vehicle(r, ego_x: float, ego_y: float, ego_v: float, role: str) -> list:
    """
    Return [dx, dy, v]
    If vehicle is missing, encode as a distant placeholder.
    """
    if r is None:
        dx = side_dx(role)

        if role in ["F", "LF", "RF"]:
            dy = +D_MAX
        elif role in ["B", "LR", "RR"]:
            dy = -D_MAX
        else:
            raise ValueError(f"Unexpected role: {role}")

        v = ego_v
    else:
        dx = float(r["Local_X"] - ego_x)
        dy = float(r["Local_Y"] - ego_y)
        v  = float(r["v_Vel"])

    return [dx, dy, v]

# ==========================================
# BUILD SAMPLES
# ==========================================
def build_samples(csv_path: str, output_path: str):
    start_time = time.time()

    print("\n" + "=" * 80)
    print("Reading CSV:", csv_path)

    df = pd.read_csv(csv_path, low_memory=False)

    needed = ["Local_X", "Local_Y", "v_Vel", LANE_COL, TIME_COL, VID_COL]
    df = df[needed].copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df.sort_values([VID_COL, TIME_COL]).reset_index(drop=True)

    print("Rows after loading:", len(df))
    print("Unique vehicles:", df[VID_COL].nunique())

    # Keep only mainline lanes if requested
    if KEEP_ONLY_MAINLINE_SAMPLES:
        df = df[df[LANE_COL].isin(MAIN_LANES)].reset_index(drop=True)
        print("Rows after keeping only mainline lanes:", len(df))

    # Group by time once
    time_groups = {t: g for t, g in df.groupby(TIME_COL, sort=False)}
    print("Time index ready. #unique times =", len(time_groups))

    X, y, meta = [], [], []

    # Iterate per vehicle
    for vid, g in df.groupby(VID_COL, sort=False):
        g = g.reset_index(drop=True)
        T = len(g)

        if T < H + F:
            continue

        lanes = g[LANE_COL].to_numpy(dtype=np.int32)
        times = g[TIME_COL].to_numpy()

        for t in range(H, T - F):
            current_lane = int(lanes[t])

            # skip sample if ego current lane is not mainline
            if KEEP_ONLY_MAINLINE_SAMPLES and current_lane not in MAIN_LANES:
                continue

            x_seq = []
            complete = True

            for k in range(t - H, t):
                ego_row = g.iloc[k]
                t_now = ego_row[TIME_COL]

                df_t = time_groups.get(t_now, None)
                if df_t is None:
                    complete = False
                    break

                ego_lane = int(ego_row[LANE_COL])

                # skip if any history point has ego outside mainline
                if KEEP_ONLY_MAINLINE_SAMPLES and ego_lane not in MAIN_LANES:
                    complete = False
                    break

                ego_x = float(ego_row["Local_X"])
                ego_y = float(ego_row["Local_Y"])
                ego_v = float(ego_row["v_Vel"])

                sur = get_surroundings(df_t, ego_row)

                frame_feat = [ego_x, ego_y, ego_v]

                for role in ["F", "B", "LF", "LR", "RF", "RR"]:
                    frame_feat += encode_vehicle(
                        sur[role], ego_x, ego_y, ego_v, role
                    )

                frame_feat += lane_flags_from_geometry(ego_lane)

                x_seq.append(frame_feat)

            if not complete or len(x_seq) != H:
                continue

            y_t = get_maneuver_label(
                current_lane=current_lane,
                future_lanes=lanes[t + 1 : t + F + 1]
            )

            X.append(x_seq)
            y.append(y_t)
            meta.append((vid, times[t]))

            if len(X) % 5000 == 0:
                print(f"Built {len(X)} samples...")

    X = np.array(X, dtype=np.float32)   # (N, 15, 23)
    y = np.array(y, dtype=np.int64)     # (N,)
    meta = np.array(meta)               # (N, 2)

    np.savez_compressed(output_path, X=X, y=y, meta=meta)

    # save one sample to CSV for inspection
    if len(X) > 0:
        feature_cols = [
            "ego_x", "ego_y", "ego_v",
            "F_dx", "F_dy", "F_v",
            "B_dx", "B_dy", "B_v",
            "LF_dx", "LF_dy", "LF_v",
            "LR_dx", "LR_dy", "LR_v",
            "RF_dx", "RF_dy", "RF_v",
            "RR_dx", "RR_dy", "RR_v",
            "CL", "CR"
        ]

        sample_idx = 0
        sample_df = pd.DataFrame(X[sample_idx], columns=feature_cols)
        sample_df["label"] = y[sample_idx]
        sample_df["Vehicle_ID"] = meta[sample_idx][0]
        sample_df["Global_Time"] = meta[sample_idx][1]

        csv_sample_path = output_path.replace(".npz", "_sample0.csv")
        sample_df.to_csv(csv_sample_path, index=False)
        print("One sample saved to CSV:", csv_sample_path)

    print("\nSaved:", output_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("meta shape:", meta.shape)
    print("Label distribution:", np.unique(y, return_counts=True))
    print("Elapsed time (min):", round((time.time() - start_time) / 60, 2))

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    build_samples(CSV_PATH, OUTPUT_NPZ)
