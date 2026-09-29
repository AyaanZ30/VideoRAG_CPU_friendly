import os
import json 
from typing import List, Any, Dict
from PIL import Image
import google.generativeai as genai

with open("src/video_search_ai/response.json", "r", encoding = 'utf-8') as file:
    RESPONSE_SCHEMA = json.load(file)

def init_gemini_client(API_KEY: str):
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")

def agent_response(query: str, retrieved_docs: List, metadata: List) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    client = init_gemini_client(API_KEY=api_key)
    
    contents = [
        "You are an AI video search assistant. You will be given a mix of "
        "AUDIO TRANSCRIPT excerpts (literal spoken/sung words) and VISUAL "
        "FRAME images (the actual picture captured at that timestamp). For "
        "questions about what is visibly happening on screen, look at the "
        "images directly and describe what you actually see -- do not rely "
        "on the audio transcript for that, since music/lyrics will not "
        "mention on-screen actions.",
        f'User Query: "{query}"',
        "Evidence:",
    ]

    for doc, meta in zip(retrieved_docs, metadata):
        start = meta.get("start", 0.0)
        end = meta.get("end", 0.0)
        video = meta.get("video", "Unknown")
        modality = meta.get("modality", "audio")

        if modality == "visual":
            frame_path = meta.get("frame_path")
            if frame_path and os.path.exists(frame_path):
                contents.append(f"[VISUAL FRAME] {video} at {start:.1f}s-{end:.1f}s:")
                contents.append(Image.open(frame_path))
            else:
                # index run predates frame_path being stored, or the file
                # got moved/deleted -- fall back rather than crash
                contents.append(f"[VISUAL FRAME - image unavailable] {video} {start:.1f}s: {doc}")
        else:
            contents.append(f"[AUDIO TRANSCRIPT] {video} at {start:.1f}s-{end:.1f}s: {doc}")

    contents.append(
      "Pick the SINGLE most relevant evidence item above that best answers "
        "the query. Respond with ONLY a raw JSON object -- no markdown code "
        "fences, no commentary before or after it -- matching exactly this "
        f"schema: {json.dumps(RESPONSE_SCHEMA)}\n"
        "'start' and 'end' must exactly match the timestamp of the evidence "
        "item you are citing, so that segment can be located and played back."
    )

    response = client.generate_content(
        contents = contents,
    )

    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw.split("\n", 1)[-1]

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"answer": response.text, "modality": None, "video": None, "start": None, "end": None}
