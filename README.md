# Asian Hornet Tracking

A computer vision pipeline for tracking Asian hornets (*Vespa velutina*) in video footage and inferring the direction of their nest. Hornets are detected with a fine-tuned YOLO model, followed by a multi-stage tracking system that bridges detection gaps with a GMM-based fallback and ultimately computes an **exit vector** this is the mean direction hornets fly when leaving the camera's field of view.

![Example output frame](image/foto_tracking.png)

---

## How it works

 
1. A fine-tuned YOLO model (`best.pt`) detects hornets in each frame.
2. BotSORT links detections into stable trajectories; when the detector loses a hornet, a MOG2 background-subtraction fallback keeps the track alive.
3. A hornet that stays still long enough is marked as *gathered* (likely foraging), filtering out hornets that merely pass through.
4. When a gathered hornet exits the frame, its trajectory is accumulated into a running mean direction vector that points toward the nest.


## Quickstart

### Requirements

```
ultralytics
opencv-python
numpy
```

### Run the tracker

```python
from tracker.tracker import Tracker, TrackerConfig

cfg = TrackerConfig(
    model_path="tracker/best.pt",
    tracker_yaml="tracker/yaml/botsort_hornets.yaml",
    target_classes=[2] #Asian Hornet
)

tracker = Tracker(cfg, verbose=True)
result = tracker.run("my_video.mp4", output_path="output.mp4", write_video=True)

print(result.exit_vector)
# ExitVector(vx=0.712, vy=-0.234, conf=0.81, n=6, angle=−18.2°)
```

### Batch processing

```python
from tracker.tracker import batch_vectors, TrackerConfig

results = batch_vectors(
    ["clip1.mp4", "clip2.mp4", "clip3.mp4"],
    cfg=TrackerConfig(),
    vector_csv="results/vectors.csv",
)
```

---

## Key configuration options

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `"best.pt"` | Path to the YOLO/RT-DETR weights file |
| `tracker_yaml` | `"botsort_hornets.yaml"` | BotSORT/Bytetrack configuration file |
| `conf_threshold` | `0.05` | YOLO detection confidence threshold |
| `use_gmm` | `True` | Enable MOG2 fallback when YOLO loses a track |
| `min_still_frames` | `50` | Frames a hornet must stay still to be considered gathered |
| `still_speed_threshold` | `4.0` | Max speed (px/frame) to count as still |
| `exit_position_weight` | `0.8` | How much the hornet's position in frame influences the exit vector vs. movement direction |
| `gmm_max_lifetime` | `40` | Max frames the GMM fallback will track a lost hornet |
| `gmm_downscale` | `0.5` | Resolution scale for background subtraction (`1.0` = full res, `0.5` = half res) |

---

## Output

`Tracker.run()` returns a `TrackResult` with:

- **`exit_vector`** — `ExitVector(vx, vy, confidence, n, angle_deg)`: the mean normalised exit direction. Higher `confidence` (0–1) means more consistent directions across hornets.
- **`predictions`** — per-frame bounding box predictions for all tracked IDs.
- **`video_path`** — path to the annotated output video (if `write_video=True`).
- **`reappearance_timestamps`** — timestamps where hornets reappeared after a gap (useful for segmenting footage).

Exit vectors are optionally appended to a CSV via the `vector_csv` parameter.

---

## Notes

- The exit vector uses **screen-space coordinates**: positive Y points down. An angle of `0°` means the hornets exit to the right, `90°` means downward, `−90°` means upward.
- The tracker is designed for **stationary camera footage** of a foraging site. It is not intended for moving cameras or dense multi-species scenes.
- GMM background subtraction is warmed up on the first `gmm_history` frames before tracking begins.