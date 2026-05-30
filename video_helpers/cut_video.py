import subprocess

ffmpeg_path = r"B:\appdir\ffmpeg\bin\ffmpeg.exe"

def cut_video(input_path, output_path, start_time, end_time):
    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-ss", start_time,
        "-to", end_time,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path
    ]

    subprocess.run(command, check=True)

# Example
cut_video(r"B:\School\Masterproef\cleaned\cropped_vids\GX010172_1280x720_full_joined.mp4", "B:\School\Masterproef\cleaned\cropped_vids\GX040171_joined_1280x720_5sec.mp4", "00:04:18", "00:4:30")