import os
import whisper 
import cv2
from typing import Any, cast

from video_db import AudioDatabase, VisualDatabase
from utils import video_file_format_check

KEYFRAME_DIR = "./keyframes"

def process_video(file_path : str):
    audio_db = AudioDatabase()
    frame_db = VisualDatabase()

    video_file = file_path.split("/")[-1]
    video_file_format_check(video_file)

    print("Loading whisper model (tiny)")
    whisper_client = whisper.load_model("tiny")
    result = whisper_client.transcribe(file_path)
    segments = result.get("segments", [])

    audio_count = 0
    if segments:
        print(f"Found audio! Indexing {len(segments)} audio segments")
        for seg in segments:
            segment = cast(dict[str, Any], seg)
            start_time = float(segment.get("start", 0.0))
            end_time = float(segment.get("end", 0.0))
            text = str(segment.get("text", ""))

            audio_db.add_segment(
                video_name = video_file,
                start = start_time,
                end = end_time,
                text = f"[Audio] {text}",
            )
            audio_count += 1

    print("Extracting visual key frames for complete multimodal coverage...")

    video_frame_dir = os.path.join(KEYFRAME_DIR, video_file)
    os.makedirs(video_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = int(fps * 5)
    frame_count = 0
    indexed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_count % sample_interval == 0:
            current_time = frame_count / fps
            frame_path = os.path.join(video_frame_dir, f"{current_time:.1f}.jpg")
            cv2.imwrite(frame_path, frame) 

            frame_db.add_keyframe(
                video_name = video_file,
                start = current_time,
                end = current_time + 5.0,
                frame_bgr = frame,
                frame_path = frame_path
            )
            indexed_frames +=1 

        frame_count += 1
    cap.release()
    print(f"Indexed {indexed_frames} visual keyframe windows successfully!")

    

