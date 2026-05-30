import subprocess
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from ultralytics import YOLO

model = YOLO(r"..\tracker\best.pt")

@dataclass
class Segment:
    start_frame: int
    end_frame: int

    def to_times(self, fps: float):
        return self.start_frame / fps, self.end_frame / fps


def _merge_segments(segments: List[Segment], pad_frames: int) -> List[Segment]:
    if not segments:
        return []
    padded = [Segment(max(0, s.start_frame - pad_frames), s.end_frame + pad_frames)
              for s in segments]
    padded.sort(key=lambda s: s.start_frame)
    merged = [padded[0]]
    for s in padded[1:]:
        if s.start_frame <= merged[-1].end_frame:
            merged[-1].end_frame = max(merged[-1].end_frame, s.end_frame)
        else:
            merged.append(s)
    return merged


def crop_video(
    video_path: str,
    output_dir: str = "clips",
    target_classes: Optional[List[int]] = None,
    pad_seconds: float = 1.0,
    min_segment_seconds: float = 0.5,
    empty_frames_trigger: int = 30,
    concat: bool = False,
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    pad_frames = int(pad_seconds * fps)
    min_frames = int(min_segment_seconds * fps)

    # ── detection pass ────────────────────────────────────────────────────
    results = model.track(
        source=video_path,
        tracker=r"..\tracker\yaml\botsort_hornets.yaml",
        persist=True,
        stream=True,
        imgsz=1280,
        classes=target_classes,
        verbose=False,
        conf=0.25
    )

    segments: List[Segment] = []
    seg_start: Optional[int] = None
    last_active_frame: int = -1

    for frame_id, result in enumerate(results):
        has_dets = (
            result.boxes is not None
            and result.boxes.id is not None
            and len(result.boxes.id) > 0
        )

        if has_dets:
            if seg_start is None:
                seg_start = frame_id
            last_active_frame = frame_id
        else:
            gap = frame_id - last_active_frame
            if seg_start is not None and gap >= empty_frames_trigger:
                seg = Segment(seg_start, last_active_frame)
                if (seg.end_frame - seg.start_frame) >= min_frames:
                    segments.append(seg)
                seg_start = None

        print(f"Scanning frame {frame_id}/{total}", end="\r")

    # close any open segment
    if seg_start is not None:
        seg = Segment(seg_start, last_active_frame)
        if (seg.end_frame - seg.start_frame) >= min_frames:
            segments.append(seg)

    segments = _merge_segments(segments, pad_frames)
    print(f"\nFound {len(segments)} active segment(s)")

    # ── cut with ffmpeg ───────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    clip_paths = []

    for i, seg in enumerate(segments):
        t_start, t_end = seg.to_times(fps)
        duration = t_end - t_start
        out_path = str(Path(output_dir) / f"{stem}_clip{i:03d}.mp4")
        ffmpeg_path = r"B:\appdir\ffmpeg\bin\ffmpeg.exe"

        subprocess.run([
            ffmpeg_path, "-y",
            "-ss", f"{t_start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            out_path,
        ], check=True, capture_output=True)

        clip_paths.append(out_path)
        print(f"  Clip {i:03d}: {t_start:.2f}s → {t_end:.2f}s  →  {out_path}")

    if concat and len(clip_paths) > 1:
        list_file = Path(output_dir) / f"{stem}_list.txt"
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{Path(p).resolve()}'\n")
        joined = Path(output_dir) / f"{stem}_joined.mp4"
        subprocess.run([
            ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(joined),
        ], check=True, capture_output=True)
        list_file.unlink()
        print(f"  Joined → {joined}")

    return clip_paths

# just people
clips = crop_video(video_path=r"B:\School\Masterproef\cleaned\cropped_vids\GX010172_1280x720_full.mp4",output_dir=r"B:\School\Masterproef\cleaned\cropped_vids", target_classes=[2],concat=True, pad_seconds=4)