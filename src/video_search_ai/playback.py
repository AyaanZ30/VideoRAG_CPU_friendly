import shutil 
import subprocess

def play_segment(video_path: str, start: float, end: float):
    if shutil.which("ffplay") is None:
        print("ffplay not found on PATH. Install ffmpeg")
        return

    duration = max(end - start, 0.1)
    cmd = [
        "ffplay", "-ss", str(start), "-t", str(duration), "-autoexit", "-loglevel", "error", video_path
    ]
    subprocess.run(cmd)