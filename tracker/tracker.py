from __future__ import annotations

import csv
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackerConfig:
    # ── required ──────────────────────────────────────────────────────────────
    model_path: str = "best.pt"

    # ── detection ─────────────────────────────────────────────────────────────
    conf_threshold: float = 0.05
    imgsz: int = 1280
    tracker_yaml: str = "botsort_hornets.yaml"
    target_classes: Optional[List[int]] = None   # None = all classes

    # ── re-link / GMM ─────────────────────────────────────────────────────────
    relink_iou_thresh: float = 0.10
    relink_dist_thresh: float = 140.0
    iou_match_threshold: float = 0.20
    lost_frames_before_gmm: int = 1
    gmm_max_lost: int = 10
    gmm_search_pad: int = 80
    gmm_min_blob_area: int = 20
    gmm_max_lifetime: int = 40      # hard cap on total GMM frames regardless of blobs

    # ── background subtractor ─────────────────────────────────────────────────
    bg_subtractor: str = "MOG2"          # "MOG2" | "LSBP" | "GSOC"
    gmm_history: int = 200
    gmm_var_threshold: float = 16.0

    # ── edge handling ─────────────────────────────────────────────────────────
    edge_margin: int = 60
    edge_search_pad: int = 50
    edge_kill_margin: int = 40

    # ── exit vector ───────────────────────────────────────────────────────────
    min_frames_for_exit: int = 15
    exit_fit_window: int = 20
    exit_position_weight: float = 0.8

    # ── drawing (only used when write_video=True) ─────────────────────────────
    draw_mode: str = "both"              # "bbox" | "track" | "both" | "none"
    track_max_len: int = 60
    track_thickness: int = 2
    track_fade: bool = True
    global_vector_draw_len: int = 80

    # ── misc ──────────────────────────────────────────────────────────────────
    fps_multiply: int = 1
    use_gmm: bool = True                 # set False to disable GMM fallback entirely
    gmm_downscale: float = 0.5   # 1.0 = original, 0.5 = half res

    # ── sparse mode ───────────────────────────────────────────────────────────
    sparse_empty_frames_trigger: int = 10   # frames with no detections before skipping starts
    sparse_ramp_every: int = 10             # every N additional empty frames, skip +1 more frame
    sparse_max_skip: int = 30              # cap on skip interval

    min_still_frames: int = 50        # frames below speed threshold to count as "gathered"
    still_speed_threshold: float = 4.0  # px/frame max speed to be considered still
    


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExitVector:
    vx: float
    vy: float
    confidence: float
    n: int

    def __repr__(self):
        angle_deg = float(np.degrees(np.arctan2(self.vy, self.vx)))
        return (f"ExitVector(vx={self.vx:.3f}, vy={self.vy:.3f}, "
                f"conf={self.confidence:.2f}, n={self.n}, "
                f"angle={angle_deg:.1f}°)")

    def as_dict(self) -> dict:
        return {
            "vx": round(self.vx, 6),
            "vy": round(self.vy, 6),
            "confidence": round(self.confidence, 6),
            "n": self.n,
            "angle_deg": round(float(np.degrees(np.arctan2(self.vy, self.vx))), 2),
        }


@dataclass
class TrackResult:
    exit_vector: ExitVector
    predictions: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    video_path: Optional[str] = None
    reappearance_timestamps:  List[dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def _center(bbox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1+x2)/2), int((y1+y2)/2))


def _dist(a, b) -> float:
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND FALLBACK TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class _BGFallback:

    def __init__(self, track_id: int, bbox_xyxy: tuple, color: tuple, cfg: TrackerConfig, cls_id=0, last_conf=1.0):
        self.track_id  = track_id
        self.color     = color
        self.bbox_xyxy = tuple(int(v) for v in bbox_xyxy)
        self.active    = True
        self._lost     = 0
        self._vx       = 0.0
        self._vy       = 0.0
        self._cfg      = cfg
        self.cls_id    = cls_id
        self.last_conf = last_conf
        self._total_frames = 0

    def _near_edge(self, h, w) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        m = self._cfg.edge_margin
        return cx < m or cx > w - m or cy < m or cy > h - m

    def _outside_frame(self, h, w) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        k = self._cfg.edge_kill_margin
        return x2 < -k or x1 > w + k or y2 < -k or y1 > h + k

    def _search_window(self, h, w):
        x1, y1, x2, y2 = self.bbox_xyxy
        p = self._cfg.edge_search_pad if self._near_edge(h, w) else self._cfg.gmm_search_pad
        return (max(0, x1-p), max(0, y1-p), min(w, x2+p), min(h, y2+p))

    def _bbox_wh(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(x2-x1, 1), max(y2-y1, 1)

    def update(self, fg_mask: np.ndarray):
        if not self.active:
            return False
        
        self._total_frames += 1
        
        if self._total_frames > self._cfg.gmm_max_lifetime:
            self.active = False
            return False
        
        h, w = fg_mask.shape
        if self._outside_frame(h, w):
            self.active = False
            return "killed"

        sx1, sy1, sx2, sy2 = self._search_window(h, w)
        roi = fg_mask[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            self._lost += 1
            return self._lost < self._cfg.gmm_max_lost

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  kernel)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        old_cx = (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2
        old_cy = (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2
        best_cnt, best_score = None, -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._cfg.gmm_min_blob_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00'] + sx1
            cy = M['m01'] / M['m00'] + sy1
            score = area / (_dist((cx, cy), (old_cx, old_cy)) + 1)
            if score > best_score:
                best_score = score; best_cnt = cnt

        bw, bh = self._bbox_wh()
        if best_cnt is not None:
            M  = cv2.moments(best_cnt)
            cx = M['m10'] / M['m00'] + sx1
            cy = M['m01'] / M['m00'] + sy1
            self._vx = 0.5*(cx-old_cx) + 0.5*self._vx
            self._vy = 0.5*(cy-old_cy) + 0.5*self._vy
            self.bbox_xyxy = (int(cx-bw/2), int(cy-bh/2),
                              int(cx+bw/2), int(cy+bh/2))
            self._lost = 0
        else:
            x1, y1, x2, y2 = self.bbox_xyxy
            self.bbox_xyxy = (int(x1+self._vx), int(y1+self._vy),
                              int(x2+self._vx), int(y2+self._vy))
            self._vx *= 0.8; self._vy *= 0.8
            self._lost += 1

        if self._lost >= self._cfg.gmm_max_lost:
            self.active = False
            return False

        return True

    def draw(self, frame: np.ndarray, track_history: dict, cfg: TrackerConfig, classid, conf: float = 1.0):
        x1, y1, x2, y2 = self.bbox_xyxy
        h, w = frame.shape[:2]
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        if x2 <= x1 or y2 <= y1:
            return
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        track_history[self.track_id].append((cx, cy))
        if cfg.draw_mode in ("bbox", "both"):
            dash = 18
            corners = [
                ((x1,y1),(x1+dash,y1)), ((x2-dash,y1),(x2,y1)),
                ((x1,y2),(x1+dash,y2)), ((x2-dash,y2),(x2,y2)),
                ((x1,y1),(x1,y1+dash)), ((x1,y2-dash),(x1,y2)),
                ((x2,y1),(x2,y1+dash)), ((x2,y2-dash),(x2,y2)),
            ]
            for pt1, pt2 in corners:
                cv2.line(frame, pt1, pt2, self.color, 2)
            label = f"ID:{self.track_id} | CLS:{classid} | {conf:.0%} [GMM {self._lost}f]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, max(0, y1-th-6)), (x1+tw, y1), (255, 255, 255), -1)
            cv2.putText(frame, label, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        if cfg.draw_mode in ("track", "both"):
            _draw_track(frame, track_history[self.track_id],
                        self.color, cfg.track_max_len,
                        cfg.track_thickness, cfg.track_fade)


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_track(frame, points, color, max_len, thickness, fade):
    if len(points) < 2:
        return
    pts = list(points)[-max_len:]
    for i in range(1, len(pts)):
        c = tuple(int(ch * i / len(pts)) for ch in color) if fade else color
        cv2.line(frame, pts[i-1], pts[i], c, thickness)


def _draw_bbox(frame, bbox, stable_id, color, classid, conf: float = 1.0):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text = f"ID:{stable_id} | CLS:{classid} | {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (x1, max(0, y1-15-th)), (x1+tw, min(frame.shape[0], y1-15+bl)),
                  (255, 255, 255), -1)
    cv2.putText(frame, text, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def _draw_global_vector(frame, gvx_sum, gvy_sum, count, draw_len):
    if count == 0:
        return
    mvx, mvy = gvx_sum / count, gvy_sum / count
    mag = float(np.hypot(mvx, mvy))
    ox, oy = 100, 100
    r = draw_len
    cv2.circle(frame, (ox, oy), r + 18, (20, 20, 20), -1)
    cv2.circle(frame, (ox, oy), r + 18, (160, 160, 160), 1)
    cv2.line(frame, (ox-r, oy), (ox+r, oy), (70, 70, 70), 1)
    cv2.line(frame, (ox, oy-r), (ox, oy+r), (70, 70, 70), 1)

    # ── Cardinal labels so you can visually verify direction ─────────────────
    label_offset = r + 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "N", (ox - 5, oy - label_offset), font, 0.4, (160, 160, 160), 1)
    cv2.putText(frame, "S", (ox - 5, oy + label_offset + 8), font, 0.4, (160, 160, 160), 1)
    cv2.putText(frame, "E", (ox + label_offset, oy + 5), font, 0.4, (160, 160, 160), 1)
    cv2.putText(frame, "W", (ox - label_offset - 12, oy + 5), font, 0.4, (160, 160, 160), 1)

    if mag > 1e-3:
        draw_len_px = int(r * mag)
        tip = (int(ox + (mvx/mag) * draw_len_px), int(oy + (mvy/mag) * draw_len_px))
        cv2.arrowedLine(frame, (ox, oy), tip, (0, 220, 255),
                        thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)

    cv2.circle(frame, (ox, oy), 4, (0, 220, 255), -1)

    angle_deg = float(np.degrees(np.arctan2(mvy, mvx)))
    # Screen-space note: Y-down means angle 90° = pointing DOWN on screen
    label = f"exit dir  n={count}  conf={mag:.2f}  {angle_deg:.0f}deg"
    cv2.putText(frame, label, (ox - r - 8, oy + r + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND SUBTRACTOR FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_bg_sub(cfg: TrackerConfig):
    kind = cfg.bg_subtractor.upper()
    if kind == "LSBP":
        try:
            return cv2.bgsegm.createBackgroundSubtractorLSBP()
        except AttributeError:
            pass
    elif kind == "GSOC":
        try:
            return cv2.bgsegm.createBackgroundSubtractorGSOC()
        except AttributeError:
            pass
    return cv2.createBackgroundSubtractorMOG2(
        history=cfg.gmm_history,
        varThreshold=cfg.gmm_var_threshold,
        detectShadows=False,
    )


def _warmup_bg_sub(bg_sub, video_path: str, n_frames: int, scale: float):
    cap = cv2.VideoCapture(video_path)
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_LINEAR
            )
        bg_sub.apply(frame)
    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  EXIT VECTOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _update_exit_vector(
    center_history,
    full_history,          # list of ALL (x, y) ever recorded for this ID
    gvx_sum, gvy_sum, count,
    stable_id: int,
    frame_w: int = 0,
    frame_h: int = 0,
    position_weight: float = 0.5,
    verbose: bool = True,
):
    all_pts = list(center_history)

    if len(all_pts) < 2:
        if verbose:
            print(f"\n  [exit vector] ID {stable_id} — SKIPPED: only {len(all_pts)} point(s) in history")
        return gvx_sum, gvy_sum, count

    first_pt = all_pts[0]
    last_pt  = all_pts[-1]
    mid_pt   = all_pts[len(all_pts) // 2]

    dx = last_pt[0] - first_pt[0]
    dy = last_pt[1] - first_pt[1]
    mov_mag = float(np.hypot(dx, dy))

    if verbose:
        print(f"\n  {'═'*60}")
        print(f"  [exit vector] ID {stable_id}  —  contributing to exit vector")
        print(f"    history: {len(all_pts)} points (deque window)")
        print(f"    first pos  : ({first_pt[0]}, {first_pt[1]})")
        print(f"    mid   pos  : ({mid_pt[0]}, {mid_pt[1]})")
        print(f"    last  pos  : ({last_pt[0]}, {last_pt[1]})")
        print(f"    net disp   : dx={dx:+.1f}  dy={dy:+.1f}  mag={mov_mag:.1f}px (deque window)")

    if mov_mag < 1e-3:
        if verbose:
            print(f"    ✗ SKIPPED — net displacement too small ({mov_mag:.3f}px)")
        return gvx_sum, gvy_sum, count

    mvx, mvy = dx / mov_mag, dy / mov_mag
    mv_angle = float(np.degrees(np.arctan2(mvy, mvx)))

    if verbose:
        print(f"    movement vec: ({mvx:+.3f}, {mvy:+.3f})  angle={mv_angle:+.1f}°  "
              f"(screen: +Y=down, so {mv_angle:+.1f}° means {'↓' if mvy>0 else '↑'}"
              f"{'→' if mvx>0 else '←'})")

    # ── Position signal ───────────────────────────────────────────────────────
    if frame_w > 0 and frame_h > 0 and position_weight > 0.0:
        ex, ey = last_pt
        px = ex - frame_w / 2.0
        py = ey - frame_h / 2.0
        pos_mag = float(np.hypot(px, py))
        if pos_mag > 1e-3:
            pvx, pvy = px / pos_mag, py / pos_mag
        else:
            pvx, pvy = mvx, mvy
            if verbose:
                print(f"    position vec: FALLBACK to movement (last pos is at frame center)")
        pv_angle = float(np.degrees(np.arctan2(pvy, pvx)))
        if verbose:
            quadrant = ("top-left" if px < 0 and py < 0 else
                        "top-right" if px > 0 and py < 0 else
                        "bottom-left" if px < 0 else "bottom-right")
            print(f"    position vec: ({pvx:+.3f}, {pvy:+.3f})  angle={pv_angle:+.1f}°")
            print(f"    last pos relative to center: ({px:+.1f}, {py:+.1f})  [{quadrant}]")
    else:
        pvx, pvy = mvx, mvy
        position_weight = 0.0
        pv_angle = mv_angle
        if verbose:
            print(f"    position vec: DISABLED (weight=0 or no frame dims)")

    # ── Blend ─────────────────────────────────────────────────────────────────
    w_mov = 1.0 - position_weight
    bx = w_mov * mvx + position_weight * pvx
    by = w_mov * mvy + position_weight * pvy
    blend_mag = float(np.hypot(bx, by))

    if blend_mag < 1e-3:
        if verbose:
            print(f"    ✗ SKIPPED — blended vector has near-zero magnitude ({blend_mag:.4f})")
        return gvx_sum, gvy_sum, count

    nx, ny = bx / blend_mag, by / blend_mag
    blend_angle = float(np.degrees(np.arctan2(ny, nx)))

    if verbose:
        print(f"    blend weights: movement={w_mov:.2f}  position={position_weight:.2f}")
        print(f"    blended vec : ({nx:+.3f}, {ny:+.3f})  angle={blend_angle:+.1f}°")
        angle_diff = abs(((blend_angle - mv_angle) + 180) % 360 - 180)
        if angle_diff > 45:
            print(f"    ⚠ WARNING: position signal pulled vector {angle_diff:.1f}° away from movement direction!")

    gvx_sum += nx
    gvy_sum += ny
    count   += 1

    mean_vx  = gvx_sum / count
    mean_vy  = gvy_sum / count
    conf     = float(np.hypot(mean_vx, mean_vy))
    mean_ang = float(np.degrees(np.arctan2(mean_vy, mean_vx)))

    if verbose:
        print(f"    ✓ ACCEPTED  → running mean=({mean_vx:+.3f}, {mean_vy:+.3f})"
              f"  angle={mean_ang:+.1f}°  conf={conf:.3f}  n={count}")
        print(f"  {'═'*60}")

    return gvx_sum, gvy_sum, count

def _build_exit_vector(gvx_sum, gvy_sum, count) -> ExitVector:
    if count == 0:
        return ExitVector(vx=0.0, vy=0.0, confidence=0.0, n=0)
    mvx, mvy = gvx_sum / count, gvy_sum / count
    conf = float(np.hypot(mvx, mvy))
    return ExitVector(vx=round(mvx, 6), vy=round(mvy, 6),
                      confidence=round(conf, 6), n=count)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRACKER CLASS
# ══════════════════════════════════════════════════════════════════════════════
class Tracker:

    def __init__(self, cfg: TrackerConfig, verbose: bool = True):
        self.cfg     = cfg
        self.verbose = verbose
        self._model  = YOLO(cfg.model_path)


    def run(
        self,
        video_path: str,
        output_path: str = "output.mp4",
        write_video: bool = True,
        vector_csv: Optional[str] = None,
    ) -> TrackResult:

        empty_frame_streak = 0
        current_skip       = 0
        skip_counter       = 0
        reappearance_timestamps: List[dict] = []
        in_sparse_mode     = False

        cfg = self.cfg

        # ── video metadata ────────────────────────────────────────────────────
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ── background subtractor ─────────────────────────────────────────────
        bg_sub = None
        if cfg.use_gmm:
            bg_sub = _make_bg_sub(cfg)
            if self.verbose:
                print(f"Warming up background model ({cfg.gmm_history} frames)…")
            _warmup_bg_sub(bg_sub, video_path, cfg.gmm_history, cfg.gmm_downscale)
            if self.verbose:
                print("  Warmup done.")
        elif self.verbose:
            print("  GMM disabled — skipping background subtractor.")

        # ── video writer ──────────────────────────────────────────────────────
        out = None
        if write_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ── state ─────────────────────────────────────────────────────────────
        track_id_colors:    dict = {}
        last_bbox:          dict = {}
        last_seen:          dict = {}
        gmm_tracks:         dict = {}
        botsort_to_stable:  dict = {}
        stable_cls:         dict = {}
        stable_confs:       dict = {}
        track_history:      dict = defaultdict(list)
        predictions:        dict = defaultdict(list)
        botsort_exited:     set  = set()
        track_still_frames: dict = defaultdict(int)
        track_gathered:     set  = set()
        full_pos_history:   dict = defaultdict(list)

        gvx_sum = gvy_sum = 0.0
        global_exit_count  = 0
        exit_center_history: dict = defaultdict(
            lambda: deque(maxlen=cfg.exit_fit_window))
        track_frame_count:   dict = defaultdict(int)

        orig_cap = cv2.VideoCapture(video_path)
        results  = self._model.track(
            source=video_path,
            tracker=cfg.tracker_yaml,
            persist=True,
            stream=True,
            imgsz=cfg.imgsz,
            conf=cfg.conf_threshold,
            verbose=False,
        )

        for frame_id, result in enumerate(results):

            # ── sparse mode: decide whether to process this frame ─────────────
            if current_skip > 0:
                skip_counter += 1
                if skip_counter < current_skip:
                    orig_cap.read()
                    continue
                else:
                    skip_counter = 0

            if cfg.fps_multiply > 1 and (frame_id % cfg.fps_multiply) != (cfg.fps_multiply - 1):
                continue

            ret, orig_frame = orig_cap.read()
            if not ret:
                break

            output_id = frame_id // cfg.fps_multiply if cfg.fps_multiply > 1 else frame_id

            # ── step 1: parse tracker detections ─────────────────────────────
            raw_dets: dict = {}
            if result.boxes is not None and result.boxes.id is not None:
                for tid, bbox, cls_id, conf in zip(
                    result.boxes.id.int().cpu().tolist(),
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.int().cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    if cfg.target_classes and cls_id not in cfg.target_classes:
                        continue
                    raw_dets[tid] = (tuple(map(int, bbox)), round(conf, 2), cls_id)

            # ── step 2: resolve tracker IDs → stable IDs ─────────────────────
            stable_dets: dict = {}
            for raw_id, (bbox, conf, cls_id) in raw_dets.items():

                if raw_id in botsort_to_stable:
                    stable_dets[botsort_to_stable[raw_id]] = (bbox, conf, cls_id)
                    continue

                bbox_c = _center(bbox)
                best_stable, best_score = None, -1

                for sid, gmm_obj in gmm_tracks.items():
                    o = _iou(bbox, gmm_obj.bbox_xyxy)
                    d = _dist(bbox_c, _center(gmm_obj.bbox_xyxy))
                    if o >= cfg.relink_iou_thresh or d <= cfg.relink_dist_thresh:
                        score = o + 1.0 / (d + 1)
                        if score > best_score:
                            best_score = score; best_stable = sid

                if best_stable is None:
                    for sid, last_b in last_bbox.items():
                        if sid in gmm_tracks:
                            continue
                        o = _iou(bbox, last_b)
                        d = _dist(bbox_c, _center(last_b))
                        if o >= cfg.relink_iou_thresh or d <= cfg.relink_dist_thresh:
                            score = o + 1.0 / (d + 1)
                            if score > best_score:
                                best_score = score; best_stable = sid

                if best_stable is not None:
                    botsort_to_stable[raw_id] = best_stable
                else:
                    botsort_to_stable[raw_id] = raw_id
                    if raw_id not in track_id_colors:
                        track_id_colors[raw_id] = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )

                stable_dets[botsort_to_stable[raw_id]] = (bbox, conf, cls_id)

            # ── step 3: bookkeeping ───────────────────────────────────────────
            for stable_id, (bbox, conf, cls_id) in stable_dets.items():
                last_seen[stable_id]    = output_id
                last_bbox[stable_id]    = bbox
                stable_cls[stable_id]   = cls_id
                stable_confs[stable_id] = conf

                cx, cy = _center(bbox)

                if stable_id not in botsort_exited:
                    exit_center_history[stable_id].append((cx, cy))
                    full_pos_history[stable_id].append((cx, cy))
                    track_frame_count[stable_id] += 1

                    if stable_id not in track_gathered:
                        pts = list(exit_center_history[stable_id])
                        if len(pts) >= 2:
                            speed = _dist(pts[-1], pts[-2])
                            if speed <= cfg.still_speed_threshold:
                                track_still_frames[stable_id] += 1
                                if track_still_frames[stable_id] >= cfg.min_still_frames:
                                    track_gathered.add(stable_id)
                                    if self.verbose:
                                        print(f"\n  [gather] ID {stable_id} GATHERED "
                                              f"at frame {output_id} "
                                              f"after {track_frame_count[stable_id]} tracked frames  "
                                              f"pos=({cx},{cy})")
                            else:
                                track_still_frames[stable_id] = 0

                if stable_id in gmm_tracks:
                    del gmm_tracks[stable_id]

            had_detections = len(stable_dets) > 0 or len(gmm_tracks) > 0
            if had_detections:
                if in_sparse_mode:
                    timestamp_sec   = output_id / fps
                    timestamp_frame = output_id
                    reappearance_timestamps.append({
                        "frame": timestamp_frame,
                        "seconds": round(timestamp_sec, 3),
                        "timecode": f"{int(timestamp_sec//60):02d}:{timestamp_sec%60:06.3f}",
                    })
                    if self.verbose:
                        print(f"\n  [sparse] reappearance at frame {timestamp_frame} "
                              f"({timestamp_sec:.3f}s) — resuming dense mode")

                empty_frame_streak = 0
                current_skip       = 0
                skip_counter       = 0
                in_sparse_mode     = False

            else:
                empty_frame_streak += 1

                if empty_frame_streak >= cfg.sparse_empty_frames_trigger:
                    frames_beyond = empty_frame_streak - cfg.sparse_empty_frames_trigger
                    new_skip = 2 + (frames_beyond // cfg.sparse_ramp_every)
                    new_skip = min(new_skip, cfg.sparse_max_skip)

                    if new_skip != current_skip:
                        current_skip   = new_skip
                        in_sparse_mode = True
                        if self.verbose:
                            print(f"\n  [sparse] streak={empty_frame_streak}  "
                                  f"skip interval → {current_skip}")

            # ── step 4: background update ─────────────────────────────────────
            fg_mask = None
            if cfg.use_gmm:
                scale = cfg.gmm_downscale
                if scale != 1.0:
                    small_frame = cv2.resize(
                        orig_frame,
                        (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    small_frame = orig_frame

                fg_mask_small = bg_sub.apply(
                    small_frame,
                    learningRate=0.0 if stable_dets else -1
                )

                if scale != 1.0:
                    fg_mask = cv2.resize(
                        fg_mask_small,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    fg_mask = fg_mask_small

            # ── step 5: spawn GMM fallbacks ───────────────────────────────────
            if cfg.use_gmm:
                for stable_id, last_frame in last_seen.items():
                    if (output_id - last_frame == cfg.lost_frames_before_gmm
                            and stable_id not in gmm_tracks
                            and stable_id not in botsort_exited
                            and stable_id in last_bbox):
                        color = track_id_colors.setdefault(stable_id, (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        ))
                        gmm_tracks[stable_id] = _BGFallback(
                            stable_id, last_bbox[stable_id], color, cfg,
                            cls_id=stable_cls.get(stable_id, 0),
                            last_conf=stable_confs.get(stable_id, 1.0),
                        )

            # ── step 6: update GMM fallbacks ──────────────────────────────────
            for stable_id in list(gmm_tracks.keys()):
                gmm_obj = gmm_tracks[stable_id]

                matched = any(
                    sid != stable_id
                    and _iou(gmm_obj.bbox_xyxy, bbox) >= cfg.iou_match_threshold
                    for sid, (bbox, conf, _cls) in stable_dets.items()
                )
                if matched:
                    del gmm_tracks[stable_id]; continue

                alive = gmm_obj.update(fg_mask)

                if alive == "killed":
                    qualifies = (
                        track_frame_count[stable_id] >= cfg.min_frames_for_exit
                        and stable_id in track_gathered
                    )
                    if self.verbose:
                        print(f"\n  [GMM killed] ID {stable_id} at frame {output_id}"
                              f"  frames={track_frame_count[stable_id]}"
                              f"  gathered={'YES' if stable_id in track_gathered else 'NO'}"
                              f"  qualifies={'YES' if qualifies else 'NO'}")
                    if qualifies:
                        gvx_sum, gvy_sum, global_exit_count = _update_exit_vector(
                            exit_center_history[stable_id],
                            full_pos_history[stable_id],
                            gvx_sum, gvy_sum, global_exit_count,
                            stable_id=stable_id,
                            frame_w=width, frame_h=height,
                            position_weight=cfg.exit_position_weight,
                            verbose=self.verbose,
                        )
                    del gmm_tracks[stable_id]; continue

                if alive is False:
                    if self.verbose:
                        print(f"\n  [GMM timeout] ID {stable_id} at frame {output_id}"
                              f"  frames={track_frame_count[stable_id]}"
                              f"  gathered={'YES' if stable_id in track_gathered else 'NO'}"
                              f"  (NOT contributing — timed out, not edge-killed)")
                    del gmm_tracks[stable_id]; continue

                last_bbox[stable_id] = gmm_obj.bbox_xyxy
                full_pos_history[stable_id].append(_center(gmm_obj.bbox_xyxy))

                if write_video:
                    gmm_obj.draw(orig_frame, track_history, cfg,
                                 gmm_obj.cls_id,
                                 getattr(gmm_obj, 'last_conf', 1.0))
                predictions[output_id + 1].append((stable_id, gmm_obj.bbox_xyxy))

            # ── step 7: draw / record YOLO stable detections ─────────────────
            for stable_id, (bbox, conf, cls_id) in stable_dets.items():
                color = track_id_colors.get(stable_id, (200, 200, 200))
                predictions[output_id + 1].append((stable_id, bbox))

                x1, y1, x2, y2 = bbox
                m = cfg.edge_kill_margin
                at_edge = (x1 < m or x2 > width - m or y1 < m or y2 > height - m)
                if at_edge and stable_id not in botsort_exited:
                    botsort_exited.add(stable_id)
                    if stable_id in gmm_tracks:
                        del gmm_tracks[stable_id]
                    qualifies = (
                        track_frame_count[stable_id] >= cfg.min_frames_for_exit
                        and stable_id in track_gathered
                    )
                    if self.verbose:
                        print(f"\n  [edge exit] ID {stable_id} at frame {output_id}"
                              f"  pos=({(x1+x2)//2},{(y1+y2)//2})"
                              f"  frames={track_frame_count[stable_id]}"
                              f"  gathered={'YES' if stable_id in track_gathered else 'NO'}"
                              f"  qualifies={'YES' if qualifies else 'NO'}")
                    if qualifies:
                        gvx_sum, gvy_sum, global_exit_count = _update_exit_vector(
                            exit_center_history[stable_id],
                            full_pos_history[stable_id],
                            gvx_sum, gvy_sum, global_exit_count,
                            stable_id=stable_id,
                            frame_w=width, frame_h=height,
                            position_weight=cfg.exit_position_weight,
                            verbose=self.verbose,
                        )

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if write_video:
                    track_history[stable_id].append((cx, cy))
                    if cfg.draw_mode in ("bbox", "both"):
                        _draw_bbox(orig_frame, bbox, stable_id, color, cls_id, conf)
                    if cfg.draw_mode in ("track", "both"):
                        _draw_track(orig_frame, track_history[stable_id],
                                    color, cfg.track_max_len,
                                    cfg.track_thickness, cfg.track_fade)

            # ── step 8: draw global exit compass ─────────────────────────────
            if write_video:
                _draw_global_vector(orig_frame, gvx_sum, gvy_sum,
                                    global_exit_count, cfg.global_vector_draw_len)
                out.write(orig_frame)

            if self.verbose:
                print(f"\rProcessed frame {output_id+1}/{total}", end="")

        orig_cap.release()
        if out is not None:
            out.release()
        if self.verbose:
            print()

        exit_vec = _build_exit_vector(gvx_sum, gvy_sum, global_exit_count)

        if self.verbose:
            print(f"\n{'═'*60}")
            print(f"  FINAL Exit vector: {exit_vec}")
            if global_exit_count == 0:
                print("  ⚠ No hornets qualified for exit vector.")
                print(f"  Check: min_frames_for_exit={cfg.min_frames_for_exit}, "
                      f"min_still_frames={cfg.min_still_frames}")
            print(f"{'═'*60}\n")

        if vector_csv:
            _write_vector_csv(vector_csv, video_path, exit_vec)

        if self.verbose:
            if write_video:
                print(f"Saved video: '{output_path}'")

        return TrackResult(
            exit_vector=exit_vec,
            predictions=predictions,
            video_path=output_path if write_video else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CSV HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _write_vector_csv(csv_path: str, video_path: str, vec: ExitVector) -> None:
    header = ["video", "vx", "vy", "confidence", "n", "angle_deg"]
    row    = [
        os.path.basename(video_path),
        vec.vx, vec.vy, vec.confidence, vec.n,
        round(float(np.degrees(np.arctan2(vec.vy, vec.vx))), 2),
    ]
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"Exit vector written to '{csv_path}'")


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: batch-run over multiple videos
# ══════════════════════════════════════════════════════════════════════════════

def batch_vectors(
    video_paths: List[str],
    cfg: TrackerConfig,
    vector_csv: Optional[str] = None,
    verbose: bool = True,
) -> List[TrackResult]:
    tracker = Tracker(cfg, verbose=verbose)
    results = []
    for vp in video_paths:
        if verbose:
            print(f"\n{'='*60}\nProcessing: {vp}")
        r = tracker.run(vp, write_video=False, vector_csv=vector_csv)
        results.append(r)
    return results