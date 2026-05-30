import copy
import csv
import itertools
import os
import random as _random
import tempfile
import time

import numpy as np
import yaml
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracker.tracker import Tracker, TrackerConfig
from ground_truth_vector import generate_gt_departure_vectors

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  ─ edit these
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_PATH_1 = r".\ground_truth\video\1.mp4"
GT_PATH_1    = r".\ground_truth\gt\1\gt\gt.txt"

VIDEO_PATH_2 = r".\ground_truth\video\2.mp4"
GT_PATH_2    = r".\ground_truth\gt\2\gt\gt.txt"

VIDEO_PATH_3 = r".\ground_truth\video\6.mp4"
GT_PATH_3    = r".\ground_truth\gt\6\gt\gt.txt"


MODEL_PATH  = r"..\tracker\best.pt"

RESULTS_CSV  = r"./csv/gt_with_vector.csv"

VIDEO_DIMS = {
    VIDEO_PATH_1: (1920, 1080),
    VIDEO_PATH_2: (1920, 1080),
    VIDEO_PATH_3: (1280, 720),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER SELECTION
# ══════════════════════════════════════════════════════════════════════════════

TRACKER_TYPE = "botsort"   # "botsort" | "bytetrack"

BASE_TRACKER_YAML = {
    "botsort":   r"..\tracker\yaml\botsort_hornets.yaml",
    "bytetrack": r"..\tracker\yaml\byterack.yaml",
}.get(TRACKER_TYPE)

_YAML_KEYS_BY_TRACKER = {
    "botsort": {
        "tracker_type", "track_high_thresh", "track_low_thresh",
        "new_track_thresh", "track_buffer", "match_thresh",
        "proximity_thresh", "appearance_thresh", "with_reid",
        "gmc_method", "fuse_score",
    },
    "bytetrack": {
        "tracker_type", "track_high_thresh", "track_low_thresh",
        "new_track_thresh", "track_buffer", "match_thresh", "fuse_score",
    },
}

# Single definition — no duplicate
_YAML_KEYS      = _YAML_KEYS_BY_TRACKER.get(TRACKER_TYPE, set())
_YAML_ONLY_KEYS = _YAML_KEYS

# ══════════════════════════════════════════════════════════════════════════════
#  DEBUG FLAG
# ══════════════════════════════════════════════════════════════════════════════

YAML_TEST_ONLY = False

# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE
# ══════════════════════════════════════════════════════════════════════════════

SEARCH_SPACE = {
    # "match_thresh": [0.1, 0.8],
    
}

SEARCH_MODE     = "grid"
N_RANDOM_TRIALS = 50
RANDOM_SEED     = 42

FIXED = {
    "conf_threshold":         0.05,
    "imgsz":                  1280,
    "relink_iou_thresh":      0.05,
    "relink_dist_thresh":     140,
    "gmm_max_lost":           20,
    "gmm_search_pad":         120,
    "gmm_min_blob_area":      10,
    "iou_match_threshold":    0.3,
    "fps_multiply":           1,
    "lost_frames_before_gmm": 1,
    "gmm_history":            200,
    "gmm_var_threshold":      16,
    "bg_subtractor":          "MOG2",
    "edge_margin":            40,
    "edge_search_pad":        50,
    "edge_kill_margin":       60,
}

EVAL_IOU_THRESHOLD = 0.5
TARGET_CLASSES     = [2]

# ══════════════════════════════════════════════════════════════════════════════
#  WARMUP FRAME CACHE
# ══════════════════════════════════════════════════════════════════════════════

import cv2

_warmup_frames: dict = {}

def _ensure_warmup_cache(video_path: str, n_frames: int) -> None:
    if video_path in _warmup_frames:
        return
    print(f"  [cache] Reading {n_frames} warmup frames from {video_path} …", flush=True)
    cap    = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    _warmup_frames[video_path] = frames
    print(f"  [cache] Cached {len(frames)} frames "
          f"({sum(f.nbytes for f in frames) / 1e6:.0f} MB)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER YAML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_TRACKER_DEFAULTS = {
    "botsort": {
        "tracker_type":    "botsort",
        "track_high_thresh": 0.5,
        "track_low_thresh":  0.1,
        "new_track_thresh":  0.6,
        "track_buffer":      30,
        "match_thresh":      0.8,
        "proximity_thresh":  0.5,
        "appearance_thresh": 0.25,
        "with_reid":         False,
        "gmc_method":        "sparseOptFlow",
        "fuse_score":        True,
    },
    "bytetrack": {
        "tracker_type":    "bytetrack",
        "track_high_thresh": 0.1,
        "track_low_thresh":  0.01,
        "new_track_thresh":  0.1,
        "track_buffer":      90,
        "match_thresh":      0.9,
        "fuse_score":        True,
    },
}


def _load_base_yaml():
    path = BASE_TRACKER_YAML
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    defaults = _TRACKER_DEFAULTS.get(TRACKER_TYPE)
    if defaults:
        return copy.deepcopy(defaults)
    return None


def write_temp_yaml(combo: dict):
    base = _load_base_yaml()
    if base is None:
        return None
    tracker_defaults = _TRACKER_DEFAULTS.get(TRACKER_TYPE, {})
    # Fill missing keys from defaults, then apply combo overrides
    for k in _YAML_KEYS:
        if k in combo:
            base[k] = combo[k]
        elif k in tracker_defaults and k not in base:
            base[k] = tracker_defaults[k]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"{TRACKER_TYPE}_trial_")
    yaml.dump(base, tmp)
    tmp.close()

    # Debug: confirm what was written
    with open(tmp.name) as f:
        contents = yaml.safe_load(f)
    print(f"    [yaml] Written to: {tmp.name}")
    print(f"    [yaml] Contents: {contents}")

    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH + METRICS
# ══════════════════════════════════════════════════════════════════════════════

def xywh_to_xyxy(x, y, w, h):
    return (x, y, x + w, y + h)


def load_ground_truth(gt_path: str) -> dict:
    gt = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts       = line.split(",")
            frame_id    = int(parts[0])
            track_id    = int(parts[1])
            x, y, w, h  = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt[frame_id].append((track_id, xywh_to_xyxy(x, y, w, h)))
    return gt


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def match_frame(gt_boxes, pred_boxes, iou_thresh):
    if not gt_boxes:
        return [], [p[0] for p in pred_boxes], [], 0.0
    if not pred_boxes:
        return [], [], [g[0] for g in gt_boxes], 0.0
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, (_, gb) in enumerate(gt_boxes):
        for j, (_, pb) in enumerate(pred_boxes):
            iou_matrix[i, j] = _iou(gb, pb)
    matched, used_gt, used_pred, iou_sum = [], set(), set(), 0.0
    flat = sorted(
        [(iou_matrix[i, j], i, j)
         for i in range(len(gt_boxes)) for j in range(len(pred_boxes))],
        reverse=True,
    )
    for iou_val, i, j in flat:
        if iou_val < iou_thresh:
            break
        if i in used_gt or j in used_pred:
            continue
        matched.append((gt_boxes[i][0], pred_boxes[j][0], iou_val))
        iou_sum += iou_val
        used_gt.add(i); used_pred.add(j)
    fp_ids = [pred_boxes[j][0] for j in range(len(pred_boxes)) if j not in used_pred]
    fn_ids = [gt_boxes[i][0]   for i in range(len(gt_boxes))   if i not in used_gt]
    return matched, fp_ids, fn_ids, iou_sum


def compute_metrics(gt: dict, predictions: dict, iou_thresh: float = 0.5) -> dict:
    all_frames       = sorted(set(gt.keys()) | set(predictions.keys()))
    total_gt = total_tp = total_fp = total_fn = total_ids = total_frag = 0
    iou_sum  = 0.0
    gt_track_frames   = defaultdict(int)
    gt_track_matched  = defaultdict(int)
    gt_track_last_pred: dict = {}
    gt_was_matched:     dict = {}
    for frame_id in all_frames:
        gt_boxes   = gt.get(frame_id, [])
        pred_boxes = predictions.get(frame_id, [])
        for gt_id, _ in gt_boxes:
            gt_track_frames[gt_id] += 1
        matches, fp_ids, fn_ids, frame_iou = match_frame(gt_boxes, pred_boxes, iou_thresh)
        total_gt  += len(gt_boxes)
        total_tp  += len(matches)
        total_fp  += len(fp_ids)
        total_fn  += len(fn_ids)
        iou_sum   += frame_iou
        for gt_id, pred_id, _ in matches:
            gt_track_matched[gt_id] += 1
            if gt_id in gt_track_last_pred and gt_track_last_pred[gt_id] != pred_id:
                total_ids += 1
            gt_track_last_pred[gt_id] = pred_id
            if gt_was_matched.get(gt_id, True) is False:
                total_frag += 1
            gt_was_matched[gt_id] = True
        for gt_id in fn_ids:
            gt_was_matched[gt_id] = False
    mota  = 1.0 - (total_fn + total_fp + total_ids) / max(total_gt, 1)
    motp  = iou_sum / max(total_tp, 1)
    n_gt  = len(gt_track_frames)
    mt    = sum(1 for tid in gt_track_frames if gt_track_matched[tid]/gt_track_frames[tid] >= 0.8)
    ml    = sum(1 for tid in gt_track_frames if gt_track_matched[tid]/gt_track_frames[tid] <= 0.2)
    idf1  = (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1)
    det_a = total_tp / max(total_tp + total_fp + total_fn, 1)
    ass_a = total_tp / max(total_tp + total_ids + total_frag, 1)
    hota  = np.sqrt(det_a * ass_a)
    return {
        "HOTA": round(hota * 100, 2),
        "MOTA": round(mota * 100, 2),
        "MOTP": round(motp * 100, 2),
        "IDF1": round(idf1 * 100, 2),
        "MT": mt, "ML": ml,
        "GT_tracks": n_gt,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "IDS":  total_ids,
        "Frag": total_frag,
        "GT_dets": total_gt,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _angle_diff(vx1, vy1, vx2, vy2) -> float:
    a1   = float(np.degrees(np.arctan2(vy1, vx1)))
    a2   = float(np.degrees(np.arctan2(vy2, vx2)))
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff


def compute_vector_metrics(gt_vec, pred_vec) -> dict:
    if gt_vec.n == 0 or pred_vec.n == 0:
        return {
            "vec_angle_diff": None,
            "vec_conf_gt":    round(gt_vec.confidence, 4),
            "vec_conf_pred":  round(pred_vec.confidence, 4),
            "vec_n_gt":       gt_vec.n,
            "vec_n_pred":     pred_vec.n,
        }
    return {
        "vec_angle_diff": round(_angle_diff(gt_vec.vx, gt_vec.vy, pred_vec.vx, pred_vec.vy), 2),
        "vec_conf_gt":    round(gt_vec.confidence, 4),
        "vec_conf_pred":  round(pred_vec.confidence, 4),
        "vec_n_gt":       gt_vec.n,
        "vec_n_pred":     pred_vec.n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COMBO → TrackerConfig BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

_CFG_FIELDS = {f.name for f in TrackerConfig.__dataclass_fields__.values()} \
    if hasattr(TrackerConfig, "__dataclass_fields__") else set()


def build_tracker_config(combo: dict) -> TrackerConfig:
    merged = copy.deepcopy(FIXED)
    merged.update({k: v for k, v in combo.items() if k not in _YAML_ONLY_KEYS})
    cfg_kwargs = {k: v for k, v in merged.items() if k in _CFG_FIELDS}
    return TrackerConfig(
        model_path=MODEL_PATH,
        target_classes=TARGET_CLASSES,
        **cfg_kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _all_combos(space: dict):
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))


def _random_combos(space: dict, n: int, seed: int):
    rng     = _random.Random(seed)
    seen    = set()
    keys    = list(space.keys())
    attempts = 0
    while len(seen) < n and attempts < n * 20:
        attempts += 1
        combo = {k: rng.choice(space[k]) for k in keys}
        key   = tuple(combo[k] for k in keys)
        if key not in seen:
            seen.add(key)
            yield combo


# ══════════════════════════════════════════════════════════════════════════════
#  CSV / RESUME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = ["HOTA", "MOTA", "MOTP", "IDF1", "MT", "ML",
                "GT_tracks", "TP", "FP", "FN", "IDS", "Frag", "GT_dets"]

_VECTOR_KEYS = ["vec_angle_diff", "vec_conf_gt", "vec_conf_pred", "vec_n_gt", "vec_n_pred"]

_SEARCH_KEYS = list(SEARCH_SPACE.keys())

_TRACKER_DEFAULTS_FOR_CSV = _TRACKER_DEFAULTS.get(TRACKER_TYPE, {})
_ALL_PARAM_KEYS = list(dict.fromkeys(
    _SEARCH_KEYS
    + list(FIXED.keys())
    + [k for k in _TRACKER_DEFAULTS_FOR_CSV if k != "tracker_type"]
))

_CSV_HEADER = (
    ["trial", "elapsed_s"]
    + _ALL_PARAM_KEYS
    + [f"gt1_{k}" for k in _METRIC_KEYS]
    + [f"gt2_{k}" for k in _METRIC_KEYS]
    + [f"gt3_{k}" for k in _METRIC_KEYS]
    + [f"gt1_{k}" for k in _VECTOR_KEYS]
    + [f"gt2_{k}" for k in _VECTOR_KEYS]
    + [f"gt3_{k}" for k in _VECTOR_KEYS]
    + ["avg_HOTA", "avg_MOTA", "avg_IDF1", "avg_vec_angle_diff"]
)


def _combo_key(combo: dict) -> tuple:
    return tuple(combo[k] for k in _SEARCH_KEYS)


def _load_completed(csv_path: str):
    completed: set = set()
    max_trial: int = 0
    best_row        = None
    best_hota       = -1.0
    if not os.path.exists(csv_path):
        return completed, max_trial, best_row
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return completed, max_trial, best_row
        col     = {name: idx for idx, name in enumerate(header)}
        missing = [k for k in _SEARCH_KEYS + ["trial", "avg_HOTA"] if k not in col]
        if missing:
            print(f"  [resume] WARNING: CSV missing columns {missing}. Starting fresh.")
            return set(), 0, None
        for row in reader:
            if not row:
                continue
            try:
                key       = tuple(type(SEARCH_SPACE[k][0])(row[col[k]]) for k in _SEARCH_KEYS)
                trial_idx = int(row[col["trial"]])
                hota      = float(row[col["avg_HOTA"]])
            except (ValueError, IndexError):
                continue
            completed.add(key)
            max_trial = max(max_trial, trial_idx)
            if hota > best_hota:
                best_hota = hota
                best_row  = dict(zip(header, row))
    return completed, max_trial, best_row


_EFFECTIVE_YAML_DEFAULTS = _load_base_yaml() or {}


def _make_row(trial_idx, elapsed, combo, m1, m2, m3, vm1, vm2, vm3):
    avg_hota    = round((m1["HOTA"] + m2["HOTA"] + m3["HOTA"]) / 3, 2)
    avg_mota    = round((m1["MOTA"] + m2["MOTA"] + m3["MOTA"]) / 3, 2)
    avg_idf1    = round((m1["IDF1"] + m2["IDF1"] + m3["IDF1"]) / 3, 2)
    angle_diffs = [v["vec_angle_diff"] for v in (vm1, vm2, vm3) if v["vec_angle_diff"] is not None]
    avg_angle   = round(sum(angle_diffs) / len(angle_diffs), 2) if angle_diffs else ""

    def _resolve(k):
        if k in combo:
            return combo[k]
        if k in FIXED:
            return FIXED[k]
        return _EFFECTIVE_YAML_DEFAULTS.get(k, "")

    return (
        [trial_idx, round(elapsed, 1)]
        + [_resolve(k) for k in _ALL_PARAM_KEYS]
        + [m1[k] for k in _METRIC_KEYS]
        + [m2[k] for k in _METRIC_KEYS]
        + [m3[k] for k in _METRIC_KEYS]
        + [vm1[k] for k in _VECTOR_KEYS]
        + [vm2[k] for k in _VECTOR_KEYS]
        + [vm3[k] for k in _VECTOR_KEYS]
        + [avg_hota, avg_mota, avg_idf1, avg_angle]
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Hornet Tracker — Hyperparameter Search")
    print("=" * 65)

    print(f"\nLoading GT 1: {GT_PATH_1}")
    gt1 = load_ground_truth(GT_PATH_1)
    print(f"  {len(gt1)} frames, {sum(len(v) for v in gt1.values())} detections")

    print(f"Loading GT 2: {GT_PATH_2}")
    gt2 = load_ground_truth(GT_PATH_2)
    print(f"  {len(gt2)} frames, {sum(len(v) for v in gt2.values())} detections")

    print(f"Loading GT 3: {GT_PATH_3}")
    gt3 = load_ground_truth(GT_PATH_3)
    print(f"  {len(gt3)} frames, {sum(len(v) for v in gt3.values())} detections")

    # ── pre-compute GT departure vectors once ─────────────────────────────────
    print("\nComputing GT departure vectors ...")
    cfg_for_gt = build_tracker_config({})
    w1, h1 = VIDEO_DIMS[VIDEO_PATH_1]
    w2, h2 = VIDEO_DIMS[VIDEO_PATH_2]
    w3, h3 = VIDEO_DIMS[VIDEO_PATH_3]

    gt_vec1 = generate_gt_departure_vectors(
        GT_PATH_1, frame_w=w1, frame_h=h1,
        edge_kill_margin=cfg_for_gt.edge_kill_margin,
        min_frames_for_exit=cfg_for_gt.min_frames_for_exit,
        exit_fit_window=cfg_for_gt.exit_fit_window,
        position_weight=cfg_for_gt.exit_position_weight,
        min_still_frames=cfg_for_gt.min_still_frames,
        still_speed_threshold=cfg_for_gt.still_speed_threshold,
    )
    gt_vec2 = generate_gt_departure_vectors(
        GT_PATH_2, frame_w=w2, frame_h=h2,
        edge_kill_margin=cfg_for_gt.edge_kill_margin,
        min_frames_for_exit=cfg_for_gt.min_frames_for_exit,
        exit_fit_window=cfg_for_gt.exit_fit_window,
        position_weight=cfg_for_gt.exit_position_weight,
        min_still_frames=cfg_for_gt.min_still_frames,
        still_speed_threshold=cfg_for_gt.still_speed_threshold,
    )
    gt_vec3 = generate_gt_departure_vectors(
        GT_PATH_3, frame_w=w3, frame_h=h3,
        edge_kill_margin=cfg_for_gt.edge_kill_margin,
        min_frames_for_exit=cfg_for_gt.min_frames_for_exit,
        exit_fit_window=cfg_for_gt.exit_fit_window,
        position_weight=cfg_for_gt.exit_position_weight,
        min_still_frames=cfg_for_gt.min_still_frames,
        still_speed_threshold=cfg_for_gt.still_speed_threshold,
    )
    print(f"  GT1 vector: {gt_vec1}")
    print(f"  GT2 vector: {gt_vec2}")
    print(f"  GT3 vector: {gt_vec3}")

    if SEARCH_MODE == "grid":
        all_trials = list(_all_combos(SEARCH_SPACE))
        print(f"\nSearch mode : GRID  ({len(all_trials)} combinations)")
    else:
        all_trials = list(_random_combos(SEARCH_SPACE, N_RANDOM_TRIALS, RANDOM_SEED))
        print(f"\nSearch mode : RANDOM  ({len(all_trials)} trials, seed={RANDOM_SEED})")

    completed, max_trial_seen, best_prev_row = _load_completed(RESULTS_CSV)
    if completed:
        print(f"\n  [resume] Found {len(completed)} completed trial(s) in {RESULTS_CSV}")
        print(f"  [resume] Continuing from trial #{max_trial_seen + 1}")

    pending = [c for c in all_trials if _combo_key(c) not in completed]
    skipped = len(all_trials) - len(pending)
    if skipped:
        print(f"  [resume] Skipping {skipped} already-done trial(s), {len(pending)} remaining.")
    if not pending:
        print("\n  All trials already completed.")
        _, _, best_prev_row = _load_completed(RESULTS_CSV)
        _print_best(best_prev_row)
        _print_results_table(RESULTS_CSV)
        return

    print(f"\nPre-loading warmup frames into RAM ...")
    _ensure_warmup_cache(VIDEO_PATH_1, FIXED["gmm_history"])
    _ensure_warmup_cache(VIDEO_PATH_2, FIXED["gmm_history"])
    _ensure_warmup_cache(VIDEO_PATH_3, FIXED["gmm_history"])

    print(f"\nEffective tracker YAML (before combo overrides):")
    print("-" * 45)
    effective = _load_base_yaml() or {}
    for k, v in effective.items():
        print(f"  {k:<26} = {v}")
    print("-" * 45)

    write_header = not os.path.exists(RESULTS_CSV)
    csv_file     = open(RESULTS_CSV, "a", newline="")
    csv_writer   = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(_CSV_HEADER)
        csv_file.flush()

    print(f"\nResults CSV : {RESULTS_CSV}")
    print(f"{'Trial':>6}  {'Elapsed':>8}  {'avgHOTA':>7}  {'avgMOTA':>7}  {'avgIDF1':>7}  {'avgAngle°':>9}")
    print("-" * 75)

    best_hota = float(best_prev_row["avg_HOTA"]) if best_prev_row else -1.0
    tmp_yamls = []
    next_idx  = max_trial_seen + 1

    try:
        for combo in pending:
            trial_idx = next_idx
            next_idx += 1

            high = combo.get("track_high_thresh", _TRACKER_DEFAULTS[TRACKER_TYPE]["track_high_thresh"])
            low  = combo.get("track_low_thresh",  _TRACKER_DEFAULTS[TRACKER_TYPE]["track_low_thresh"])
            if low >= high:
                print(f"  Trial {trial_idx}: SKIPPED — "
                      f"track_low_thresh ({low}) >= track_high_thresh ({high})")
                next_idx -= 1
                continue

            t0        = time.time()
            yaml_path = write_temp_yaml(combo)
            if yaml_path:
                tmp_yamls.append(yaml_path)

            if YAML_TEST_ONLY:
                print(f"\n  [YAML TEST] Written to: {yaml_path}")
                with open(yaml_path) as f:
                    print(f.read())
                break

            cfg = build_tracker_config(combo)
            if yaml_path:
                cfg.tracker_yaml = yaml_path
            tracker = Tracker(cfg, verbose=False)

            print(f"\n  Trial {trial_idx}/{max_trial_seen + len(all_trials)}  params={combo}")

            print(f"    [1/3] Tracking video 1 ...", flush=True)
            result1 = tracker.run(VIDEO_PATH_1, write_video=False)
            print(f"    [1/3] Done  ({time.time()-t0:.1f}s)", flush=True)

            print(f"    [2/3] Tracking video 2 ...", flush=True)
            result2 = tracker.run(VIDEO_PATH_2, write_video=False)
            print(f"    [2/3] Done  ({time.time()-t0:.1f}s)", flush=True)

            print(f"    [3/3] Tracking video 3 ...", flush=True)
            result3 = tracker.run(VIDEO_PATH_3, write_video=False)
            print(f"    [3/3] Done  ({time.time()-t0:.1f}s)  Computing metrics ...", flush=True)

            m1 = compute_metrics(gt1, result1.predictions, EVAL_IOU_THRESHOLD)
            m2 = compute_metrics(gt2, result2.predictions, EVAL_IOU_THRESHOLD)
            m3 = compute_metrics(gt3, result3.predictions, EVAL_IOU_THRESHOLD)

            vm1 = compute_vector_metrics(gt_vec1, result1.exit_vector)
            vm2 = compute_vector_metrics(gt_vec2, result2.exit_vector)
            vm3 = compute_vector_metrics(gt_vec3, result3.exit_vector)

            elapsed  = time.time() - t0
            avg_hota = (m1["HOTA"] + m2["HOTA"] + m3["HOTA"]) / 3
            avg_mota = (m1["MOTA"] + m2["MOTA"] + m3["MOTA"]) / 3
            avg_idf1 = (m1["IDF1"] + m2["IDF1"] + m3["IDF1"]) / 3
            angle_diffs = [v["vec_angle_diff"] for v in (vm1, vm2, vm3) if v["vec_angle_diff"] is not None]
            avg_angle   = sum(angle_diffs) / len(angle_diffs) if angle_diffs else float("nan")

            csv_writer.writerow(_make_row(trial_idx, elapsed, combo, m1, m2, m3, vm1, vm2, vm3))
            csv_file.flush()

            flag = ""
            if avg_hota > best_hota:
                best_hota = avg_hota
                flag = "  ◀ best"

            print(f"{trial_idx:>6}  {elapsed:>7.1f}s  "
                  f"{avg_hota:>7.2f}  {avg_mota:>7.2f}  {avg_idf1:>7.2f}  "
                  f"{avg_angle:>9.2f}{flag}")

    finally:
        csv_file.close()
        for yp in tmp_yamls:
            try:
                os.unlink(yp)
            except OSError:
                pass

    _, _, global_best_row = _load_completed(RESULTS_CSV)
    _print_best(global_best_row)
    _print_results_table(RESULTS_CSV)
    print(f"\nFull results saved to: {RESULTS_CSV}")


def _print_results_table(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("avg_HOTA")]
    if not rows:
        return
    rows.sort(key=lambda r: float(r["avg_HOTA"]), reverse=True)
    print("\n" + "=" * 110)
    print("  ALL TRIALS (sorted by avg HOTA)")
    print("=" * 110)
    print(f"  {'#':>5}  {'avgHOTA':>7}  {'avgMOTA':>7}  {'avgIDF1':>7}  {'avgAngle°':>9}  "
          + "  ".join(f"{k[:14]:<14}" for k in _SEARCH_KEYS))
    print("-" * 110)
    for r in rows:
        param_vals = "  ".join(f"{r.get(k,'?'):<14}" for k in _SEARCH_KEYS)
        avg_angle  = r.get("avg_vec_angle_diff", "?")
        print(f"  {r['trial']:>5}  {float(r['avg_HOTA']):>7.2f}  "
              f"{float(r['avg_MOTA']):>7.2f}  {float(r['avg_IDF1']):>7.2f}  "
              f"{avg_angle:>9}  {param_vals}")
    print("=" * 110)


def _print_best(best_row) -> None:
    if best_row is None:
        print("\nNo completed trials to summarise.")
        return
    print("\n" + "=" * 65)
    print(f"  BEST TRIAL: #{best_row.get('trial', '?')}   "
          f"avg HOTA = {float(best_row['avg_HOTA']):.2f} %")
    print("=" * 65)
    print("\n  Parameters:")
    for k in _SEARCH_KEYS:
        print(f"    {k:<26} = {best_row.get(k, '?')}")
    print(f"\n  avg vector angle diff: {best_row.get('avg_vec_angle_diff', '?')}°")
    for gt_label in ("gt1", "gt2", "gt3"):
        print(f"\n  {gt_label.upper()} metrics:")
        for k in ["HOTA", "MOTA", "MOTP", "IDF1", "TP", "FP", "FN", "IDS", "Frag"]:
            print(f"    {k:<20} {best_row.get(f'{gt_label}_{k}', '?')}")
        print(f"    {'vec_angle_diff':<20} {best_row.get(f'{gt_label}_vec_angle_diff', '?')}°")


if __name__ == "__main__":
    main()