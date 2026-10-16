"""
Context-Aware Video Captioning - FastAPI backend.

This file contains the ENTIRE pipeline in one place, on purpose,
so it is easy to read top to bottom and easy to explain in a viva.

Captioning pipeline:
    video file
      -> OpenCV frame sampling
      -> each sampled frame sent to Qwen-VL (through Ollama) for a
         short visual observation
      -> audio track transcribed with faster-whisper (pretrained,
         runs on CPU so it doesn't compete with Qwen for GPU/VRAM)
      -> visual observations + transcript combined and sent to Qwen,
         which reasons over both and produces one context-aware caption
      -> caption is turned into an embedding and stored, along with
         the video file, in a simple local JSON "database"

Search pipeline:
    user types a query
      -> query turned into an embedding (same model as captions)
      -> compared against every stored caption embedding with
         cosine similarity
      -> best match above a threshold is returned, otherwise
         "no matching video found"

No PyTorch/TensorFlow API is used directly anywhere in this file.
Qwen runs entirely inside Ollama. faster-whisper and
sentence-transformers do pull in PyTorch as an internal dependency
(see requirements.txt / PROJECT_GUIDE.md for details), but we only
ever call their simple Python APIs - never torch directly.
"""

import base64
import glob
import json
import os
import tempfile

import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Ollama's local REST API. Ollama must already be running
# (it starts automatically once installed on Windows).
OLLAMA_URL = "http://localhost:11434/api/chat"

# Vision-language model pulled through Ollama.
# See PROJECT_GUIDE.md for why this specific model was chosen.
VLM_MODEL = "qwen3-vl:8b"

# We never process every frame of the video - that would be slow and
# mostly redundant, since neighbouring frames look almost identical.
MIN_FRAMES = 5
MAX_FRAMES = 15

# Where uploaded videos and the "database" file live.
UPLOAD_DIR = "."
DB_PATH = "video_database.json"

# Cosine similarity a search query must reach to count as a match.
# 0.6 is a common practical middle ground for MiniLM-style embeddings:
# related-but-differently-worded sentences ("playing football" vs
# "kicking a ball on a field") typically score 0.6-0.8, while
# unrelated sentences usually fall below 0.4. See PROJECT_GUIDE.md.
SIMILARITY_THRESHOLD = 0.5


app = FastAPI(title="Context-Aware Video Captioning")



# Loaded once at startup and reused for every request.
# - Whisper runs on CPU with int8 so it doesn't compete with Qwen
#   for the laptop's 8 GB of VRAM.
# - MiniLM is tiny (~80 MB) and runs instantly on CPU.
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------
# Step 1: Frame sampling with OpenCV
# ---------------------------------------------------------------------

def sample_frames(video_path: str):
    """
    Pick a small number of frames spread evenly across the video
    instead of reading every single frame.

    A short video gets fewer frames, a longer one gets more,
    capped between MIN_FRAMES and MAX_FRAMES so Qwen never has to
    look at too many images (that would be slow and use a lot of
    VRAM/time for no real benefit).
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps else 0

    # Roughly one sampled frame per second of video, clamped to a
    # sensible range so we never send too few or too many frames.
    num_frames = int(duration_seconds)
    num_frames = max(MIN_FRAMES, min(MAX_FRAMES, num_frames))

    # Evenly spaced frame indices across the whole video.
    frame_indices = [
        int(i * total_frames / num_frames) for i in range(num_frames)
    ]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)

    cap.release()
    return frames


def frame_to_base64_jpg(frame) -> str:
    """Encode an OpenCV (numpy) frame as a base64 JPEG string,
    which is the image format Ollama's chat API expects."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Could not encode frame as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------
# Step 2: Ask Qwen-VL to describe a single frame
# ---------------------------------------------------------------------

def describe_frame(frame) -> str:
    """
    Send one frame to Qwen-VL and ask for a short, factual
    description of what is visible in ONLY this frame.

    We deliberately keep this prompt narrow (no guessing about
    "what happens next") - the reasoning across frames happens
    later, in build_final_caption().
    """
    image_b64 = frame_to_base64_jpg(frame)

    prompt = (
        "Describe only what is visually present in this single image "
        "in one short sentence. Mention people, objects, and actions "
        "you can actually see. Do not guess what happened before or "
        "after this image."
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ---------------------------------------------------------------------
# Step 3: Audio -> text with faster-whisper
# ---------------------------------------------------------------------

def transcribe_audio(video_path: str) -> str:
    """
    Transcribe the speech in a video file using a pretrained
    faster-whisper model.

    faster-whisper can read the audio track directly out of a video
    file, so there's no separate "extract audio" step in our code.
    If the video has no audio, or anything goes wrong, we fall back
    to an empty transcript so the rest of the pipeline still works
    using visual information only.
    """
    try:
        segments, _ = whisper_model.transcribe(video_path, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Step 4: Chronological aggregation + context reasoning
# ---------------------------------------------------------------------

def build_final_caption(observations: list[str], transcript: str) -> str:
    """
    Combine the per-frame visual observations (in chronological order)
    with the audio transcript, and ask Qwen to reason about the whole
    event - not just repeat each observation or the transcript. This
    is the step that turns disconnected sentences into one coherent,
    2-3 sentence context-aware caption.
    """
    numbered_observations = "\n".join(
        f"Frame {i + 1}: {obs}" for i, obs in enumerate(observations)
    )

    reasoning_prompt = (
        "VISUAL OBSERVATIONS (in chronological order):\n"
        f"{numbered_observations}\n\n"
        "AUDIO TRANSCRIPT (what was said during the video):\n"
        f"{transcript if transcript else '(no speech detected)'}\n\n"
        "Using the visual observations AND the audio transcript together, "
        "write a 2-3 sentence description of what happened across the "
        "video. Follow these rules:\n"
        "- Describe events in chronological order.\n"
        "- Preserve important actions and events, including brief ones.\n"
        "- Mention notable interactions between people and objects.\n"
        "- Use the transcript when it adds useful context or intention, "
        "but do not repeat it word for word.\n"
        "- Connect events into one coherent description rather than "
        "listing frames separately.\n"
        "- Never invent details that aren't supported by the "
        "observations or transcript.\n"
        "- Avoid unnecessary analysis or explanation - just describe "
        "what happened, concisely."
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": reasoning_prompt}],
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ---------------------------------------------------------------------
# Step 5: Caption -> embedding, and the local video "database"
# ---------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Turn a piece of text (a caption or a search query) into a
    vector using a small pretrained sentence-embedding model."""
    return embedder.encode(text).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_database() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        return json.load(f)


def save_database(db: list[dict]) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def next_video_id() -> str:
    """A simple incrementing ID like video_001, video_002,"""
    db = load_database()
    return f"video_{len(db) + 1:03d}"

def store_video(tmp_path: str, original_filename: str, caption: str) -> dict:
    """Store the original video path, caption, and embedding."""
    video_id = next_video_id()
    
    record = {
        "video_id": video_id,
        "video_path": stored_path,
        "caption": caption,
        "embedding": embed_text(caption),
    }
    db = load_database()
    db.append(record)
    save_database(db)
    return record

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    db = load_database()

    for record in db:
        if record["video_id"] == video_id:
            video_path = record["video_path"]

            if not os.path.exists(video_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": "Video file not found."}
                )

            return FileResponse(video_path)

    return JSONResponse(
        status_code=404,
        content={"error": "Video ID not found."}
    )


# ---------------------------------------------------------------------
# Step 6: Full captioning pipeline
# ---------------------------------------------------------------------

def generate_caption(video_path: str):
    """Run the full pipeline on a video file and return the final
    caption plus how many frames were analyzed."""
    frames = sample_frames(video_path)

    # Frame-level observations, kept in the same order as the video.
    observations = [describe_frame(frame) for frame in frames]

    transcript = transcribe_audio(video_path)

    final_caption = build_final_caption(observations, transcript)

    return final_caption, len(frames)


# ---------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------

@app.post("/caption")
async def caption_video(video: UploadFile = File(...)):
    """
    Accepts an uploaded video file, runs it through the pipeline,
    stores the video + caption + embedding in the local database,
    and returns the final caption as JSON:

        { "caption": "...", "frames_analyzed": 8 }
    """
    # Save the upload to a temporary file because OpenCV/whisper need
    # a real file path, not a stream of bytes.
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        caption, frames_analyzed = generate_caption(tmp_path)
        store_video(tmp_path, video.filename, caption)
    except requests.exceptions.ConnectionError:
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    "Could not reach Ollama. Make sure Ollama is running "
                    f"and that '{VLM_MODEL}' has been pulled."
                )
            },
        )
    finally:
        # Always clean up the temporary file, even if something failed.
        os.remove(tmp_path)

    return {"caption": caption, "frames_analyzed": frames_analyzed}


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
from fastapi.responses import FileResponse
async def search_video(request: SearchRequest):
    """
    Semantic search over previously captioned videos.

    Embeds the query, compares it with every stored caption embedding
    using cosine similarity, and returns the best match if it clears
    SIMILARITY_THRESHOLD.
    """
    db = load_database()
    if not db:
        return {"found": False, "message": "No matching video found in the database."}

    query_embedding = embed_text(request.query)
    scored = [(record, cosine_similarity(query_embedding, record["embedding"])) for record in db]
    best_record, best_score = max(scored, key=lambda pair: pair[1])

    if best_score < SIMILARITY_THRESHOLD:
        return {"found": False, "message": "No matching video found in the database."}

    return {
        "found": True,
        "video_id": best_record["video_id"],
        "video_path": f"/video/{best_record['video_id']}",
        "caption": best_record["caption"],
        "similarity": round(best_score, 2),
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the one-page frontend so no separate frontend server
    is needed."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
