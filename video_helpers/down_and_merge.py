import subprocess
import os
import glob

ffmpeg_path = r"B:\appdir\ffmpeg\bin\ffmpeg.exe"

def resize_video(input_path, output_path, width, height):
    command = [
        ffmpeg_path,
        "-i", input_path,
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vsync", "vfr",
        output_path
    ]
    subprocess.run(command, check=True)

def merge_videos(video_list, output_path):
    # Create a temporary file list for ffmpeg concat
    list_file = os.path.join(os.path.dirname(output_path), "_concat_list.txt")
    with open(list_file, "w") as f:
        for v in video_list:
            # ffmpeg requires forward slashes or escaped backslashes
            f.write(f"file '{v.replace(chr(92), '/')}'\n")

    command = [
        ffmpeg_path,
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    os.remove(list_file)

def process_directory(input_dir, output_dir, width=1280, height=720):
    os.makedirs(output_dir, exist_ok=True)

    # Find all video files (add/remove extensions as needed)
    extensions = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    video_files.sort()  # Sort for consistent merge order

    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video(s):")
    for v in video_files:
        print(f"  {v}")

    # Downscale each video
    resized_files = []
    for input_path in video_files:
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_{width}x{height}.mp4")
        print(f"\nResizing: {input_path} -> {output_path}")
        resize_video(input_path, output_path, width, height)
        resized_files.append(output_path)

    # Merge all resized videos
    merged_output = os.path.join(output_dir, "merged_output.mp4")
    print(f"\nMerging {len(resized_files)} video(s) into: {merged_output}")
    merge_videos(resized_files, merged_output)
    print("\nDone!")

# --- Configure these paths ---
INPUT_DIR  = r"H:\nuttig"
OUTPUT_DIR = r"H:\nuttig\output"
WIDTH, HEIGHT = 1280, 720

process_directory(INPUT_DIR, OUTPUT_DIR, WIDTH, HEIGHT)