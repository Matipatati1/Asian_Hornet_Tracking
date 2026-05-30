import subprocess
import os
import glob

ffmpeg_path = r"B:\appdir\ffmpeg\bin\ffmpeg.exe"

def merge_videos(video_list, output_path):
    # Create temporary concat list
    list_file = os.path.join(os.path.dirname(output_path), "_concat_list.txt")

    with open(list_file, "w", encoding="utf-8") as f:
        for video in video_list:
            # Convert backslashes to forward slashes for ffmpeg
            f.write(f"file '{video.replace(chr(92), '/')}'\n")

    command = [
        ffmpeg_path,
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]

    subprocess.run(command, check=True)

    # Remove temp file
    os.remove(list_file)

def process_directory(input_dir, output_file):
    # Video extensions
    video_files = []

    extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]

    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    # Remove duplicates + sort
    video_files = sorted(set(video_files))

    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video(s):")

    for v in video_files:
        print(f"  {v}")

    print(f"\nMerging into: {output_file}")

    merge_videos(video_files, output_file)

    print("\nDone!")

# --- Configure paths ---
INPUT_DIR = r"H:\nuttig\output\need_merge"
OUTPUT_FILE = r"H:\nuttig\merged_output_juist.mp4"

process_directory(INPUT_DIR, OUTPUT_FILE)