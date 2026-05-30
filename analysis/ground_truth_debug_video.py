import cv2
import numpy as np
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracker.tracker import Tracker, TrackerConfig

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  ─ edit these
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_PATH       = r"B:\School\Masterproef\Tracking\video\Cut long_cut_001.mp4"
GT_PATH          = r"B:\School\Masterproef\Tracking\ground_truth\gt\2\gt\gt.txt"
DEBUG_VIDEO_PATH = r"B:\School\Masterproef\cleaned\results\eval_debug_bytertrack.mp4"

_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(_HERE, "..", "tracker", "best.pt")
TRACKER_YAML = os.path.join(_HERE, "..", "tracker", "yaml", "botsort_hornets.yaml")


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER CONFIG  ─ edit these to match the parameters you want to visualise
# ══════════════════════════════════════════════════════════════════════════════

cfg = TrackerConfig(
    model_path             = MODEL_PATH,
    tracker_yaml           = TRACKER_YAML,
    target_classes         = [2],
    conf_threshold         = 0.05,
    imgsz                  = 1280,
    lost_frames_before_gmm = 1,
    gmm_max_lost           = 20,
    gmm_search_pad         = 120,
    gmm_history            = 200,
    gmm_var_threshold      = 16,
    gmm_min_blob_area      = 10,
    iou_match_threshold    = 0.3,
    bg_subtractor          = "MOG2",
    relink_iou_thresh      = 0.05,
    relink_dist_thresh     = 140,
    edge_margin            = 40,
    edge_search_pad        = 30,
    edge_kill_margin       = 60,
)

# ── evaluation ────────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5   # minimum IoU to count as a true positive

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1

COLOR_TP  = (0,   200,   0)   # green  — true positive
COLOR_FP  = (0,     0, 220)   # red    — false positive
COLOR_FN  = (0,   140, 255)   # orange — false negative
COLOR_GT  = (0,   230,   0)   # green  — ground truth panel
COLOR_DIV = (255, 255, 255)   # white divider


def _draw_solid_box(img, bbox, color, label="", thickness=2):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), bl = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        by1 = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, by1), (x1 + tw + 2, y1), color, -1)
        lum = int(0.299*color[2] + 0.587*color[1] + 0.114*color[0])
        tc  = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 1, y1 - 3), _FONT, _FONT_SCALE, tc, _FONT_THICK)


def _draw_dashed_box(img, bbox, color, label="", thickness=2, dash=12):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    for p1, p2 in [
        ((x1, y1), (x1+dash, y1)), ((x2-dash, y1), (x2, y1)),
        ((x1, y2), (x1+dash, y2)), ((x2-dash, y2), (x2, y2)),
        ((x1, y1), (x1, y1+dash)), ((x1, y2-dash), (x1, y2)),
        ((x2, y1), (x2, y1+dash)), ((x2, y2-dash), (x2, y2)),
    ]:
        cv2.line(img, p1, p2, color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        by1 = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, by1), (x1 + tw + 2, y1), color, -1)
        lum = int(0.299*color[2] + 0.587*color[1] + 0.114*color[0])
        tc  = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 1, y1 - 3), _FONT, _FONT_SCALE, tc, _FONT_THICK)


def _overlay_text(img, lines, x=6, y=20, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * 18),
                    _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK + 2)
        cv2.putText(img, line, (x, y + i * 18),
                    _FONT, _FONT_SCALE, color, _FONT_THICK)


def build_debug_frame(orig_frame, gt_boxes, pred_boxes, matches, fp_ids, fn_ids, frame_id):
    """Return a side-by-side BGR image (GT left, predictions right)."""
    h, w  = orig_frame.shape[:2]
    left  = orig_frame.copy()
    right = orig_frame.copy()

    # ── left panel: ground truth ──────────────────────────────────────────────
    for gt_id, bbox in gt_boxes:
        _draw_solid_box(left, bbox, COLOR_GT, label=f"GT {gt_id}")
    _overlay_text(left, [f"Frame {frame_id}", f"GT boxes: {len(gt_boxes)}"])

    # ── right panel: predictions ──────────────────────────────────────────────
    matched_pred_ids = {pred_id for _, pred_id, _ in matches}
    pred_to_gt       = {pred_id: gt_id for gt_id, pred_id, _ in matches}

    for pred_id, bbox in pred_boxes:
        if pred_id in matched_pred_ids:
            _draw_solid_box(right, bbox, COLOR_TP,
                            label=f"P{pred_id}↔GT{pred_to_gt[pred_id]}")
        else:
            _draw_solid_box(right, bbox, COLOR_FP, label=f"FP {pred_id}")

    fn_id_set = set(fn_ids)
    for gt_id, bbox in gt_boxes:
        if gt_id in fn_id_set:
            _draw_dashed_box(right, bbox, COLOR_FN, label=f"FN GT{gt_id}")

    _overlay_text(right, [
        f"Frame {frame_id}",
        f"TP {len(matches)}  FP {len(fp_ids)}  FN {len(fn_ids)}",
    ])

    # ── combine ───────────────────────────────────────────────────────────────
    combined = np.hstack([left, right])
    cv2.line(combined, (w, 0), (w, h), COLOR_DIV, 2)
    cv2.putText(combined, "GROUND TRUTH",
                (w // 2 - 70, h - 10), _FONT, 0.65, COLOR_DIV, 2)
    cv2.putText(combined, "TRACKER  (G=TP  R=FP  O=FN)",
                (w + w // 2 - 140, h - 10), _FONT, 0.65, COLOR_DIV, 2)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH LOADER
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
            parts    = line.split(",")
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt[frame_id].append((track_id, xywh_to_xyxy(x, y, w, h)))
    return gt


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME MATCHING
# ══════════════════════════════════════════════════════════════════════════════

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
    """Greedy IoU matching. Returns (matches, fp_ids, fn_ids, iou_sum)."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Hornet Tracker — Debug Video")
    print("=" * 55)

    # ── load ground truth ─────────────────────────────────────────────────────
    print(f"\nLoading ground truth: {GT_PATH}")
    gt       = load_ground_truth(GT_PATH)
    gt_dets  = sum(len(v) for v in gt.values())
    print(f"  {len(gt)} frames, {gt_dets} total GT detections")

    # ── run tracker (no video) ────────────────────────────────────────────────
    print(f"\nRunning tracker on: {VIDEO_PATH}")
    result     = Tracker(cfg).run(VIDEO_PATH, write_video=False)
    predictions = result.predictions
    pred_dets  = sum(len(v) for v in predictions.values())
    print(f"  {len(predictions)} frames with predictions, {pred_dets} total detections")

    # ── set up debug video writer ─────────────────────────────────────────────
    src_cap = cv2.VideoCapture(VIDEO_PATH)
    fps     = src_cap.get(cv2.CAP_PROP_FPS)
    src_w   = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(DEBUG_VIDEO_PATH, fourcc, fps, (src_w * 2, src_h))
    print(f"\nWriting debug video: {DEBUG_VIDEO_PATH}")
    print(f"  Resolution: {src_w*2} × {src_h}  @ {fps:.1f} fps")

    # ── render frame by frame ─────────────────────────────────────────────────
    all_frames = sorted(set(gt.keys()) | set(predictions.keys()))
    n_frames   = max(all_frames) if all_frames else 0

    for frame_id in all_frames:
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, orig = src_cap.read()
        if not ret:
            continue

        gt_boxes   = gt.get(frame_id, [])
        pred_boxes = predictions.get(frame_id, [])
        matches, fp_ids, fn_ids, _ = match_frame(gt_boxes, pred_boxes, IOU_THRESHOLD)

        debug_frame = build_debug_frame(
            orig, gt_boxes, pred_boxes, matches, fp_ids, fn_ids, frame_id)
        writer.write(debug_frame)
        print(f"\r  Frame {frame_id}/{n_frames}", end="", flush=True)

    src_cap.release()
    writer.release()
    print(f"\n\nDebug video saved: '{DEBUG_VIDEO_PATH}'")
