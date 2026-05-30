import argparse
import os
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import csv

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Install ultralytics: pip install ultralytics")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

VIDEO_PATH  = r"B:\School\Masterproef\cleaned\cropped_vids\YTDown_Shorts_European-hornet-on-flowering-ivy_Media_yAObMAuxalg_001_1080p.mp4"
MODEL_PATH  = r"..\tracker\best.pt"
TRACKER_YAML = r"..\tracker\yaml\botsort_hornets.yaml"

TARGET_CLASS = 1
CONF_THRESHOLD = 0.25
IMGSZ = 1280
DEVICE = ""

SWEEP_STEPS = 20
CSV_PATH = "./confidence_sweep_neiuwe_europe.csv"


# ─────────────────────────────────────────────────────────────
# COLLECT DETECTIONS (run once)
# ─────────────────────────────────────────────────────────────

def collect_raw_detections(video_path, model_path, tracker_yaml, imgsz, device):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")

    tracker_cfg = tracker_yaml if os.path.exists(tracker_yaml) else "botsort.yaml"

    raw = defaultdict(list)

    kwargs = dict(
        source=video_path,
        tracker=tracker_cfg,
        imgsz=imgsz,
        stream=True,
        conf=0.01,
        verbose=False,
    )

    if device:
        kwargs["device"] = device

    t0 = time.time()

    for frame_idx, result in enumerate(model.track(**kwargs)):
        if result.boxes is None:
            continue

        for i in range(len(result.boxes)):
            raw[frame_idx].append({
                "cls": int(result.boxes.cls[i].item()),
                "conf": float(result.boxes.conf[i].item()),
            })

        if frame_idx % 200 == 0:
            print(f"Frame {frame_idx}/{total_frames}")

    print(f"Tracking done in {time.time() - t0:.1f}s")
    return raw, total_frames


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_class_errors(raw, total_frames, target_class, conf_thresh):
    total_dets = 0
    correct_dets = 0
    wrong_dets = 0

    for dets in raw.values():
        for d in dets:
            if d["conf"] < conf_thresh:
                continue

            total_dets += 1

            if d["cls"] == target_class:
                correct_dets += 1
            else:
                wrong_dets += 1

    accuracy = correct_dets / total_dets if total_dets > 0 else 0
    error_rate = wrong_dets / total_dets if total_dets > 0 else 0

    return {
        "threshold": round(conf_thresh, 3),
        "total_dets": total_dets,
        "correct": correct_dets,
        "wrong": wrong_dets,
        "accuracy": round(accuracy * 100, 2),
        "error_rate": round(error_rate * 100, 2),
    }


# ─────────────────────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────────────────────

def sweep_thresholds(raw, total_frames, target_class, steps):
    thresholds = np.linspace(0.01, 0.95, steps)
    results = []

    for t in thresholds:
        r = compute_class_errors(raw, total_frames, target_class, float(t))
        results.append(r)

    return results


# ─────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────

def save_csv(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "total_dets", "correct", "wrong", "accuracy", "error_rate"])

        for r in results:
            writer.writerow([
                r["threshold"],
                r["total_dets"],
                r["correct"],
                r["wrong"],
                r["accuracy"],
                r["error_rate"],
            ])

    print(f"Saved CSV → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(video_path: Optional[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tracker", default=None)
    parser.add_argument("--steps", type=int, default=SWEEP_STEPS)
    parser.add_argument("--csv", default=CSV_PATH)
    args = parser.parse_args()

    video = args.video or video_path or VIDEO_PATH
    model = args.model or MODEL_PATH
    tracker = args.tracker or TRACKER_YAML

    print("=" * 60)
    print("Confidence Sweep — Wrong Class Analysis")
    print("=" * 60)

    # 1. run once
    raw, total_frames = collect_raw_detections(
        video, model, tracker, IMGSZ, DEVICE
    )

    # 2. sweep
    results = sweep_thresholds(raw, total_frames, TARGET_CLASS, args.steps)

    # 3. print table
    print("\nthreshold | total | wrong | error %")
    print("-" * 40)

    for r in results:
        print(f"{r['threshold']:<9} | {r['total_dets']:<5} | {r['wrong']:<5} | {r['error_rate']:<7}")

    # 4. save
    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()