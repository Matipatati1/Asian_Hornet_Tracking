from __future__ import annotations

import csv
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


import tracker as tracker_mod
from tracker import Tracker, TrackerConfig, _update_exit_vector


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class HornetVectors:
    __slots__ = ("stable_id", "mvx", "mvy", "pvx", "pvy", "mov_mag")

    def __init__(self, stable_id: int,
                 mvx: float, mvy: float,
                 pvx: float, pvy: float,
                 mov_mag: float):
        self.stable_id = stable_id
        self.mvx, self.mvy = mvx, mvy   # unit movement vector
        self.pvx, self.pvy = pvx, pvy   # unit position vector
        self.mov_mag = mov_mag           # raw displacement magnitude (px)

    def blend(self, position_weight: float) -> Tuple[float, float]:
        """Return the normalised blended (nx, ny) for a given weight."""
        w_mov = 1.0 - position_weight
        bx = w_mov * self.mvx + position_weight * self.pvx
        by = w_mov * self.mvy + position_weight * self.pvy
        mag = float(np.hypot(bx, by))
        if mag < 1e-3:
            return 0.0, 0.0
        return bx / mag, by / mag


# ══════════════════════════════════════════════════════════════════════════════
#  MONKEY-PATCH: intercept raw vectors before blending
# ══════════════════════════════════════════════════════════════════════════════

class _RawVectorCollector:

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.hornets: List[HornetVectors] = []

    def patched_update(
        self,
        center_history, full_history,
        gvx_sum, gvy_sum, count,
        stable_id, frame_w=0, frame_h=0,
        position_weight=0.5, verbose=False,
    ):
        # ── replicate the qualification logic from _update_exit_vector ────────
        all_pts = list(center_history)
        if len(all_pts) < 2:
            return gvx_sum, gvy_sum, count

        first_pt, last_pt = all_pts[0], all_pts[-1]
        dx = last_pt[0] - first_pt[0]
        dy = last_pt[1] - first_pt[1]
        mov_mag = float(np.hypot(dx, dy))
        if mov_mag < 1e-3:
            return gvx_sum, gvy_sum, count

        # unit movement vector
        mvx, mvy = dx / mov_mag, dy / mov_mag

        # unit position vector
        fw = frame_w or self.frame_w
        fh = frame_h or self.frame_h
        if fw > 0 and fh > 0:
            px = last_pt[0] - fw / 2.0
            py = last_pt[1] - fh / 2.0
            pos_mag = float(np.hypot(px, py))
            if pos_mag > 1e-3:
                pvx, pvy = px / pos_mag, py / pos_mag
            else:
                pvx, pvy = mvx, mvy
        else:
            pvx, pvy = mvx, mvy

        # store raw vectors — do NOT update gvx_sum/count here
        self.hornets.append(HornetVectors(stable_id, mvx, mvy, pvx, pvy, mov_mag))

        # still need to return updated sums so the tracker's internal state
        # stays consistent (it uses the return value to draw the live compass)
        # → blend with the requested weight and add to sums
        nx, ny = HornetVectors(stable_id, mvx, mvy, pvx, pvy, mov_mag).blend(position_weight)
        if abs(nx) < 1e-9 and abs(ny) < 1e-9:
            return gvx_sum, gvy_sum, count
        return gvx_sum + nx, gvy_sum + ny, count + 1


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _circular_std_deg(vecs: List[Tuple[float, float]]) -> float:
    if len(vecs) < 2:
        return 0.0
    angles = [np.arctan2(vy, vx) for vx, vy in vecs]
    R = float(np.hypot(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    return round(float(np.degrees(np.sqrt(-2.0 * np.log(min(R, 1.0) + 1e-9)))), 2)


def _sweep_weights(
    hornets: List[HornetVectors],
    weights: List[float],
    tracker_label: str,
    gmm_label: str,
) -> List[dict]:
    rows = []
    for w in weights:
        blended = [h.blend(w) for h in hornets]
        valid   = [(vx, vy) for vx, vy in blended if abs(vx) > 1e-9 or abs(vy) > 1e-9]
        n = len(valid)
        if n == 0:
            rows.append({
                "tracker": tracker_label, "gmm": gmm_label,
                "weight": round(w, 3), "confidence": 0.0, "n": 0,
                "angle_deg": float("nan"), "circ_std_deg": 0.0,
                "vx": 0.0, "vy": 0.0,
            })
            continue

        mean_vx = sum(vx for vx, _ in valid) / n
        mean_vy = sum(vy for _, vy in valid) / n
        conf    = float(np.hypot(mean_vx, mean_vy))
        angle   = float(np.degrees(np.arctan2(mean_vy, mean_vx)))

        rows.append({
            "tracker":      tracker_label,
            "gmm":          gmm_label,
            "weight":       round(w, 3),
            "confidence":   round(conf, 4),
            "n":            n,
            "angle_deg":    round(angle, 2),
            "circ_std_deg": _circular_std_deg(valid),
            "vx":           round(mean_vx, 6),
            "vy":           round(mean_vy, 6),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE RUN  (called once per gmm/tracker combo)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    video_path: str,
    use_gmm: bool,
    tracker_yaml: str,
    base_cfg: TrackerConfig,
    write_video: bool = False,
    run_tag: str = "",
) -> List[HornetVectors]:
    cfg = deepcopy(base_cfg)
    cfg.use_gmm      = use_gmm
    cfg.tracker_yaml = tracker_yaml
    cfg.exit_position_weight = 0.5   # arbitrary — collector captures raw vecs

    import cv2
    cap    = cv2.VideoCapture(video_path)
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    collector = _RawVectorCollector(fw, fh)
    original  = tracker_mod._update_exit_vector
    tracker_mod._update_exit_vector = collector.patched_update

    try:
        t = Tracker(cfg, verbose=False)
        t.run(
            video_path,
            output_path=f"_eval_{run_tag}.mp4",
            write_video=write_video,
            vector_csv=None,
        )
    finally:
        tracker_mod._update_exit_vector = original

    return collector.hornets


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_HDR = (
    f"  {'Rank':<4} {'Tracker':<22} {'GMM':<6} "
    f"{'Weight':<8} {'Conf':<9} {'n':<5} "
    f"{'Angle':>9}  {'CircStd':>8}"
)

def _row(r: dict, rank: int, is_best: bool) -> str:
    marker = "  ◄ BEST" if is_best else ""
    return (
        f"  {rank:<4} {r['tracker']:<22} {r['gmm']:<6} "
        f"{r['weight']:<8.2f} {r['confidence']:<9.4f} {r['n']:<5} "
        f"{r['angle_deg']:>9.2f}°  {r['circ_std_deg']:>7.2f}°{marker}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ══════════════════════════════════════════════════════════════════════════
    #  PARAMS — edit these, then just run:  python eval_position_weight.py
    # ══════════════════════════════════════════════════════════════════════════

    VIDEO             = "your_footage.mp4"

    # Weight grid to evaluate (0.0 = pure movement, 1.0 = pure position)
    WEIGHTS           = [round(x * 0.1, 1) for x in range(11)]   # 0.0 → 1.0

    TRACKERS          = ["botsort", "bytetrack"]

    # GMM fallback "on", "off", or "both"
    GMM               = "both"

    OUT_CSV           = "weight_eval_results.csv"

    MODEL             = "best.pt"
    BOTSORT_YAML      = "botsort_hornets.yaml"
    BYTETRACK_YAML    = "bytetrack.yaml"

    CONF              = 0.05
    IMGSZ             = 1280

    WRITE_VIDEO       = False

    # ══════════════════════════════════════════════════════════════════════════

    if not Path(VIDEO).exists():
        print(f"ERROR: video not found: {VIDEO}")
        sys.exit(1)

    gmm_flags = (
        [True, False] if GMM == "both"
        else [True]   if GMM == "on"
        else [False]
    )
    tracker_yaml_map = {
        "botsort":   BOTSORT_YAML,
        "bytetrack": BYTETRACK_YAML,
    }
    tracker_yamls = [(t, tracker_yaml_map[t]) for t in TRACKERS]
    weights       = sorted(set(WEIGHTS))

    pipeline_combos = list(product(gmm_flags, tracker_yamls))
    n_pipelines     = len(pipeline_combos)
    n_weight_evals  = len(weights) * n_pipelines
    total_old       = len(weights) * len(gmm_flags) * len(tracker_yamls)

    print(f"\n{'═'*65}")
    print(f"  Efficient grid search")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  Pipeline runs needed : {n_pipelines}  "
          f"(vs {total_old} with naive approach)")
    print(f"  Weight evaluations   : {n_weight_evals}  (pure Python, ~instant)")
    print(f"  Weights to test      : {weights}")
    print(f"  Video : {VIDEO}")
    print(f"{'═'*65}\n")

    base_cfg = TrackerConfig(
        model_path=MODEL,
        conf_threshold=CONF,
        imgsz=IMGSZ,
    )

    all_rows: List[dict] = []

    for run_idx, (use_gmm, (tracker_name, yaml_path)) in enumerate(pipeline_combos, 1):
        gmm_label     = "on" if use_gmm else "off"
        tracker_label = Path(yaml_path).stem
        run_tag       = f"gmm{gmm_label}_{tracker_label}"

        if not Path(yaml_path).exists():
            print(f"  [{run_idx}/{n_pipelines}] SKIP — yaml not found: {yaml_path}")
            continue

        print(f"  [{run_idx}/{n_pipelines}] Running pipeline: "
              f"GMM={gmm_label}  tracker={tracker_label} …")

        try:
            hornets = run_pipeline(
                VIDEO, use_gmm, yaml_path, base_cfg,
                write_video=WRITE_VIDEO,
                run_tag=run_tag,
            )
        except Exception as exc:
            print(f"    ERROR during pipeline run: {exc}")
            continue

        print(f"    → {len(hornets)} qualifying hornet(s) captured. "
              f"Sweeping {len(weights)} weights …")

        rows = _sweep_weights(hornets, weights, tracker_label, gmm_label)
        all_rows.extend(rows)

        # quick per-run summary
        best_here = max(rows, key=lambda r: (r["confidence"], r["n"]))
        print(f"    Best weight for this combo: "
              f"{best_here['weight']}  "
              f"conf={best_here['confidence']:.4f}  "
              f"n={best_here['n']}")

    if not all_rows:
        print("\nNo results — check your yaml paths and video.")
        sys.exit(1)

    # ── write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ["tracker", "gmm", "weight", "confidence", "n",
                  "angle_deg", "circ_std_deg", "vx", "vy"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Results saved → {OUT_CSV}")

    # ── rank and print ────────────────────────────────────────────────────────
    ranked = sorted(all_rows, key=lambda r: (r["confidence"], r["n"]), reverse=True)
    best   = ranked[0]

    print(f"\n{'═'*65}")
    print(f"  FULL RANKING")
    print(f"  {'-'*63}")
    print(_HDR)
    print(f"  {'-'*63}")
    for i, r in enumerate(ranked, 1):
        print(_row(r, i, i == 1))

    # best per tracker
    print(f"\n{'═'*65}")
    print(f"  BEST PER TRACKER")
    print(f"  {'-'*63}")
    print(_HDR)
    print(f"  {'-'*63}")
    seen: set = set()
    for r in ranked:
        if r["tracker"] not in seen:
            seen.add(r["tracker"])
            print(_row(r, ranked.index(r) + 1, r is best))

    # best per GMM
    print(f"\n{'═'*65}")
    print(f"  BEST PER GMM SETTING")
    print(f"  {'-'*63}")
    print(_HDR)
    print(f"  {'-'*63}")
    seen = set()
    for r in ranked:
        if r["gmm"] not in seen:
            seen.add(r["gmm"])
            print(_row(r, ranked.index(r) + 1, r is best))

    # overall winner
    print(f"\n{'═'*65}")
    print(f"  BEST CONFIG OVERALL")
    print(f"  {'─'*63}")
    print(f"    tracker              : {best['tracker']}")
    print(f"    GMM                  : {best['gmm']}")
    print(f"    exit_position_weight : {best['weight']}")
    print(f"    confidence           : {best['confidence']:.4f}  (1.0 = perfect agreement)")
    print(f"    hornets (n)          : {best['n']}")
    print(f"    exit angle           : {best['angle_deg']:+.2f}°")
    print(f"    circ std             : {best['circ_std_deg']:.2f}°  (lower = more consistent)")
    print(f"{'═'*65}\n")

    # impact notes
    valid = [r for r in all_rows if r["n"] > 0]
    if valid:
        confs = [r["confidence"] for r in valid]
        if max(confs) - min(confs) < 0.05:
            print("  NOTE: confidence spread < 0.05 across all combos — "
                  "parameter choice has little impact on this footage.\n")

        for axis, key in [("GMM", "gmm"), ("Tracker", "tracker")]:
            groups: Dict[str, list] = {}
            for r in valid:
                groups.setdefault(r[key], []).append(r["confidence"])
            if len(groups) == 2:
                labels = list(groups)
                means  = {k: float(np.mean(v)) for k, v in groups.items()}
                better = max(means, key=means.__getitem__)
                delta  = abs(means[labels[0]] - means[labels[1]])
                print(f"  {axis} impact: avg conf delta = {delta:.4f}  "
                      f"({better} is generally better on this footage)")
        print()


if __name__ == "__main__":
    main()