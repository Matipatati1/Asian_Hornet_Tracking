import subprocess

ffmpeg_path = r"B:\appdir\ffmpeg\bin\ffmpeg.exe"

def resize_video(input_path, output_path, width, height):
    command = [
        ffmpeg_path,
        "-i", input_path,
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",        # use 'fast' since you're downscaling 4K, saves time
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vsync", "vfr",          # handles GoPro's variable frame rate
        output_path
    ]
    subprocess.run(command, check=True)

resize_video(
    r"H:\DCIM\100GOPRO\GX010177.MP4",
    r"B:\School\Masterproef\cleaned\cropped_vids\GX010177_1280x720_full.mp4",
    1280, 720
)