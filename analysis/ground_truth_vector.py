# gt_vectors.py
from __future__ import annotations

import csv
from collections import defaultdict, deque
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from tracker.tracker import (
    ExitVector,
    _build_exit_vector,
    _update_exit_vector,
    _write_vector_csv,
)


def generate_gt_departure_vectors(
    mot_path: str,
    frame_w: int,
    frame_h: int,
    edge_kill_margin: int = 40,
    min_frames_for_exit: int = 15,
    exit_fit_window: int = 20,
    position_weight: float = 0.8,
    min_still_frames: int = 8,
    still_speed_threshold: float = 8.0,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> ExitVector:

    # ── load MOT annotations ──────────────────────────────────────────────────
    tracks = defaultdict(list)
    with open(mot_path, newline="") as f:
        for row in csv.reader(f):
            frame_id, tid = int(row[0]), int(row[1])
            x, y, w, h    = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            cx, cy         = x + w / 2, y + h / 2
            tracks[tid].append((frame_id, cx, cy))

    for tid in tracks:
        tracks[tid].sort(key=lambda t: t[0])

    gvx_sum = gvy_sum = 0.0
    count = 0
    per_track_rows = []

    skipped_too_short  = []
    skipped_not_gathered = []
    skipped_not_at_edge  = []
    accepted = []

    for tid, detections in tracks.items():

        # ── gate 1: minimum track length ──────────────────────────────────────
        if len(detections) < min_frames_for_exit:
            skipped_too_short.append(tid)
            continue

        # ── gate 2: must have gathered (paused) at some point ─────────────────
        gathered       = False
        still_streak   = 0
        gathered_frame = None
        for i in range(1, len(detections)):
            _, cx1, cy1 = detections[i - 1]
            _, cx2, cy2 = detections[i]
            speed = float(np.hypot(cx2 - cx1, cy2 - cy1))
            if speed <= still_speed_threshold:
                still_streak += 1
                if still_streak >= min_still_frames:
                    gathered       = True
                    gathered_frame = detections[i][0]
                    break
            else:
                still_streak = 0

        if not gathered:
            skipped_not_gathered.append(tid)
            continue

        # ── gate 3: last detection must be at the frame edge ──────────────────
        exit_frame, last_cx, last_cy = detections[-1]
        at_edge = (
            last_cx < edge_kill_margin or
            last_cx > frame_w - edge_kill_margin or
            last_cy < edge_kill_margin or
            last_cy > frame_h - edge_kill_margin
        )
        if not at_edge:
            skipped_not_at_edge.append(tid)
            continue

        # ── build histories ───────────────────────────────────────────────────
        # full_history  — all positions, used for movement direction (matches tracker)
        # rolling deque — last exit_fit_window positions (passed as center_history)
        full_history    = [(cx, cy) for _, cx, cy in detections]
        rolling_history = deque(
            [(cx, cy) for _, cx, cy in detections[-exit_fit_window:]],
            maxlen=exit_fit_window,
        )

        # ── contribute to global vector ───────────────────────────────────────
        gvx_sum, gvy_sum, count = _update_exit_vector(
            center_history=rolling_history,
            full_history=full_history,
            gvx_sum=gvx_sum,
            gvy_sum=gvy_sum,
            count=count,
            stable_id=tid,
            frame_w=frame_w,
            frame_h=frame_h,
            position_weight=position_weight,
            verbose=verbose,
        )

        # ── per-track vector (recompute for this track alone) ─────────────────
        tvx, tvy, tcount = _update_exit_vector(
            center_history=rolling_history,
            full_history=full_history,
            gvx_sum=0.0,
            gvy_sum=0.0,
            count=0,
            stable_id=tid,
            frame_w=frame_w,
            frame_h=frame_h,
            position_weight=position_weight,
            verbose=False,   # already printed above
        )
        track_vec = _build_exit_vector(tvx, tvy, tcount)

        per_track_rows.append({
            "track_id":       tid,
            "exit_frame":     exit_frame,
            "gathered_frame": gathered_frame,
            "exit_cx":        round(last_cx, 2),
            "exit_cy":        round(last_cy, 2),
            "total_frames":   len(detections),
            "vx":             round(track_vec.vx, 6),
            "vy":             round(track_vec.vy, 6),
            "angle_deg":      round(float(np.degrees(np.arctan2(track_vec.vy, track_vec.vx))), 2),
            "confidence":     round(track_vec.confidence, 6),
        })
        accepted.append(tid)

    # ── summary print ─────────────────────────────────────────────────────────
    if verbose:
        total_tracks = len(tracks)
        print(f"\n{'═'*60}")
        print(f"  GT vector summary")
        print(f"  Total tracks          : {total_tracks}")
        print(f"  Accepted              : {len(accepted)}  {accepted}")
        print(f"  Skipped (too short)   : {len(skipped_too_short)}  {skipped_too_short}")
        print(f"  Skipped (not gathered): {len(skipped_not_gathered)}  {skipped_not_gathered}")
        print(f"  Skipped (not at edge) : {len(skipped_not_at_edge)}  {skipped_not_at_edge}")
        print(f"{'═'*60}\n")

    result = _build_exit_vector(gvx_sum, gvy_sum, count)

    # ── write CSV ─────────────────────────────────────────────────────────────
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            fieldnames = [
                "track_id", "exit_frame", "gathered_frame",
                "exit_cx", "exit_cy", "total_frames",
                "vx", "vy", "angle_deg", "confidence",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_track_rows)

            f.write("\n")
            summary = csv.writer(f)
            summary.writerow(["# SUMMARY", "vx", "vy", "confidence", "n", "angle_deg"])
            summary.writerow([
                "mean",
                round(result.vx, 6),
                round(result.vy, 6),
                round(result.confidence, 6),
                result.n,
                round(float(np.degrees(np.arctan2(result.vy, result.vx))), 2),
            ])

        print(f"GT vectors written to '{output_csv}'")

    print(f"GT departure vector: {result}")
    return result


gt_vec = generate_gt_departure_vectors(
    mot_path=r".\ground_truth\gt\6\gt\gt.txt",
    frame_w=1280,
    frame_h=720,
    edge_kill_margin=40,
    min_frames_for_exit=15,
    exit_fit_window=20,
    position_weight=0.8,
    min_still_frames=50,
    still_speed_threshold=4.0,
    output_csv=r".\csv\gt6_vectors.csv",
    verbose=True,
)


# compare against tracker output
# from tracker import Tracker, TrackerConfig
#
# cfg = TrackerConfig(model_path=r"B:\School\Masterproef\cleaned\models\best.pt")
# tracker = Tracker(cfg)
# result = tracker.run("video.mp4", write_video=False)
#
# print(f"GT:      {gt_vec}")
# print(f"Tracker: {result.exit_vector}")
#
# angle_diff = abs(
#     np.degrees(np.arctan2(gt_vec.vy, gt_vec.vx)) -
#     np.degrees(np.arctan2(result.exit_vector.vy, result.exit_vector.vx))
# )
# print(f"Angle difference: {angle_diff:.1f}°")