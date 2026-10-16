# Context-Aware Video Captioning — Project Guide

## 1. Project Overview

This project takes a short video and produces **one** natural-language
caption that describes what happened across the *whole* video, using
both what it **sees** and what it **hears**. It uses a pretrained
vision-language model (Qwen-VL, through Ollama) and a pretrained
speech-to-text model (faster-whisper) together, then stores each
caption as a sentence embedding so later videos can be found again
with a plain-English search.

## 2. Problem Being Solved

Most beginner-level captioning projects (including my earlier Flickr8k
project) work on a **single static image**. Videos are different: a
single frame rarely tells the whole story, and a lot of what's
happening is only clear from what people *say*, not just what they
*do*. A useful video caption has to capture change over time, and
combine visual and spoken information the way a person naturally
would when describing a clip.

## 3. Why Ordinary Image Captioning Is Insufficient

If you caption every frame independently and glue the captions
together, you get something disconnected and repetitive, e.g. "A
person standing near a table." "A person holding a phone." "A phone is
on the table." This doesn't explain *why* things happened, doesn't
use what was said out loud, and doesn't connect events into a story.

## 4. What "Context-Aware" Means in This Project

"Context-aware" means the final caption comes from a **reasoning
step** that looks at all the frame observations *and* the audio
transcript *together*, in order, and infers the overall event — the
way a person would describe a clip after watching and listening to
it, rather than describing disconnected screenshots. The final output
is 2-3 connected sentences (not just one), so it can preserve
important actions, brief but notable events, and object interactions
in chronological order — while still being concise.

## 5. Complete Architecture

```
VIDEO
 │
 ├── Visual
 │     ↓
 │   OpenCV frame sampling
 │     ↓
 │   Qwen-VL
 │     ↓
 │   Visual observations
 │
 └── Audio
       ↓
     faster-whisper
       ↓
     Transcript

Visual observations + transcript
 ↓
Qwen context reasoning
 ↓
Final caption
 ↓
Sentence embedding
 ↓
Local video database (video_database.json + videos/)
```

Search, once videos are stored:

```
User query
 ↓
Query embedding
 ↓
Cosine similarity vs. every stored caption embedding
 ↓
Matching caption/video (or "no match" if below threshold)
 ↓
Display + play video
```

## 6. Complete Data Flow

1. User uploads a video through the web page (or points the notebook
   at a local video file).
2. OpenCV samples a small number of evenly spaced frames (roughly
   1 frame per second, between 5 and 15 frames).
3. Each frame is sent to Qwen-VL with a narrow prompt asking only for
   what's visible in that one frame.
4. In parallel (conceptually — the code just runs it as a separate
   step), the video file is handed to faster-whisper, which extracts
   and transcribes any speech in the audio track.
5. The numbered visual observations and the transcript are combined
   into one reasoning prompt sent to Qwen, clearly labeled as
   "VISUAL OBSERVATIONS" and "AUDIO TRANSCRIPT".
6. Qwen returns one final, context-aware caption.
7. The caption is embedded with a small sentence-embedding model, and
   the video file + caption + embedding are saved locally.
8. Later, a user can type a natural-language query. It's embedded the
   same way, compared against every stored caption embedding with
   cosine similarity, and the best match (if good enough) is returned
   and played back in the browser.

## 7. What Qwen Does

Qwen-VL "looks" at the frames and also does the final reasoning. It's
used in two roles: describing individual frames, and reasoning over
the combined visual + audio information to produce one sentence. It
is never fine-tuned — we only send it prompts through Ollama.

## 8. What Whisper (faster-whisper) Does

faster-whisper is a pretrained speech-to-text model. It listens to the
audio track of the video and converts spoken words into text — nothing
more. It does not identify speakers, detect emotion, or classify
sounds; it only produces a transcript. If a video has no speech (or no
audio track at all), the transcript is simply left empty and the
pipeline continues using visual information alone.

## 9. What OpenCV Does

OpenCV (`cv2`) is used only for standard video handling: opening a
video file, reading its FPS/frame count, reading specific frames by
index, and encoding a frame as JPEG so it can be sent to Qwen. It does
no "understanding" — all visual and audio understanding comes from
Qwen and Whisper respectively.

## 10. What the Custom Pipeline Does

This is the actual contribution of the project:

- Deciding how many frames to sample and which ones.
- Prompting Qwen for focused, per-frame visual observations.
- Running speech-to-text and handling the case where it fails or
  finds no speech.
- Combining visual and audio information into one clearly-labeled
  reasoning prompt.
- Turning the final caption into an embedding and managing a simple
  local video database.
- Implementing semantic search over that database.
- Exposing all of this through a small API and a one-page frontend.

## 11. How Frame Sampling Works

The video's duration in seconds is used as a rough guide for how many
frames to sample (about one per second), clamped between 5 and 15.
Frame indices are spread evenly across the whole video so both short
and long videos are covered start to end.

## 12. How Temporal/Context Reasoning Works

Reasoning is done entirely through prompting — no separate reasoning
model. The frame observations are numbered in chronological order and
the transcript is placed underneath, clearly labeled. Qwen is asked,
in plain language, to explain how things changed over time using
*both* sources of information, so the final caption isn't just a
frame-by-frame list or a raw copy of the transcript.

## 13. Why We Don't Process Every Video Frame

Consecutive frames are almost identical, so sending every frame to
Qwen would be slow and mostly redundant. Sampling a handful of
representative frames captures the important changes while keeping
the pipeline fast enough for a laptop GPU.

## 14. Why We Don't Train Qwen or Whisper

Training or fine-tuning either model would need large labeled datasets
and heavy compute — impractical for an 8 GB laptop GPU and a one-week
timeline. Both models are used exactly as published, and the
engineering effort goes into the pipeline around them instead.

## 15. Why PyTorch/TensorFlow Are NOT Required (and one honest caveat)

Qwen runs entirely inside Ollama, which our code only ever talks to
over plain HTTP — no PyTorch involved there at all.

**Caveat, stated plainly:** `sentence-transformers` (used for caption
and query embeddings) declares PyTorch as one of its own internal
dependencies, so `pip install sentence-transformers` will pull in
`torch` automatically. This project's code never imports `torch` or
calls any PyTorch API directly — we only ever call
`SentenceTransformer.encode(...)`. `faster-whisper`, by contrast, does
**not** use PyTorch at all; it runs on CTranslate2 (a separate C++
inference engine), which is one of the reasons it was chosen over
plain `openai-whisper` (see section 16).

## 16. Why faster-whisper Was Chosen

Two well-known pretrained Whisper implementations were compared:

- **openai-whisper** — the original reference implementation, built
  directly on PyTorch.
- **faster-whisper** — a reimplementation of the same pretrained
  Whisper weights using CTranslate2, a lightweight C++ inference
  engine, instead of PyTorch.

faster-whisper was chosen because it does the same job with a smaller
dependency footprint (no PyTorch pulled in for the audio side), lower
memory use, and faster CPU inference — all useful on a laptop that's
already using its GPU for Qwen. The Python API is also very simple
(`model.transcribe(path)`), which keeps the code easy to read.

## 17. What Embeddings Are

An embedding is just a list of numbers (a vector) that represents the
*meaning* of a piece of text. Sentences with similar meaning end up
with vectors that point in similar directions, even if they use
completely different words. This is what lets "a person playing
football" and "someone kicking a ball on a field" be recognized as
related, without any hand-written rules.

## 18. Why We Embed Captions, Not Raw Video

Comparing raw video frames directly would need a much heavier
vision-embedding pipeline and wouldn't map naturally onto free-text
search queries. Since we've already asked Qwen to summarize the video
as one clear sentence, embedding *that sentence* gives us a compact,
meaningful representation that's directly comparable to a user's
typed query — using a small, fast, well-understood text model.

## 19. What Cosine Similarity Does

Cosine similarity measures how similar two vectors' *directions* are,
ignoring their length. It returns a score between -1 and 1 (in
practice, close to 0-1 for normal sentences): 1 means "pointing in
exactly the same direction" (very similar meaning), and lower values
mean less related meaning. It's the standard, simple way to compare
two embeddings.

## 20. How the Local Database Works

There is no database server. `video_database.json` is a plain JSON
file containing a list of records:

```json
[
  {
    "video_id": "video_001",
    "video_path": "videos/video_001.mp4",
    "caption": "A person is playing football in a field.",
    "embedding": [0.0123, -0.045, ...]
  }
]
```

Videos themselves live in a `videos/` folder, and FastAPI serves that
folder as static files at `/videos/...`, so the frontend can play them
back with a normal `<video>` tag.

## 21. How Semantic Search Works

1. The user's query is embedded with the same MiniLM model used for
   captions.
2. Cosine similarity is computed between the query embedding and
   *every* stored caption embedding (a simple loop — practical for a
   handful to a few hundred videos).
3. The highest-scoring video is picked.
4. If its similarity is at or above `SIMILARITY_THRESHOLD` (0.6), it's
   returned as a match; otherwise the API reports no match found.

## 22. How the Video Is Retrieved

Once a matching record is found, its `video_path` (e.g.
`videos/video_001.mp4`) is converted into a URL the browser can use
(`/videos/video_001.mp4`), because FastAPI mounts the `videos/` folder
as static files. The frontend sets that URL as the `src` of a
`<video controls>` element so the user can play it directly.

## 23. Why Regex Is NOT Used

Regex can only match exact words or patterns you explicitly write
rules for. It has no notion of *meaning*, so "playing football" and
"kicking a ball on a field" would not match at all under regex, even
though they describe the same thing. Embeddings + cosine similarity
solve exactly this problem by comparing meaning instead of exact text.

## 24. Why We Don't Need MongoDB / a Vector Database Here

With a handful to a few hundred videos (realistic for a one-week
student project), comparing a query embedding against every stored
embedding in a simple Python loop takes a fraction of a second. A
vector database (FAISS, Chroma, Pinecone, etc.) exists to make that
comparison fast at *millions* of records — solving a scaling problem
this project doesn't have. A single JSON file keeps the whole system
easy to inspect, back up, and explain.

## 25. What Happens When No Matching Video Exists

If the database is empty, or the best similarity score is below the
threshold (0.6), the `/search` endpoint returns
`{"found": false, "message": "No matching video found in the database."}`,
and the frontend displays that message instead of a video player.

## 26. How Audio Improves Context

Some information simply isn't visible — intentions, names, reasons,
or plans someone states out loud ("I am preparing vegetables for
dinner"). Adding the transcript alongside the visual observations lets
Qwen produce a caption that reflects *why* something is happening, not
just *what* is visually happening, when that information was spoken
during the video.

## 27. Limitations

- Works best on short, simple videos (roughly under a minute) with a
  small number of clear actions and speakers.
- Best suited to everyday activities, not fast, complex, or crowded
  scenes.
- Frame sampling can miss very brief actions between sampled frames.
- Transcription quality depends on audio clarity; background noise or
  heavy accents can reduce accuracy (Whisper's "base" model is small
  and fast, not the most accurate available).
- Semantic search compares against captions only — it can't find
  details that never made it into the final caption.
- Cosine-similarity search checks every stored video in a simple loop,
  which is fine for a small personal collection but wouldn't scale to
  a huge video library without a real vector index.

## 28. Possible Future Improvements

- Adaptive frame sampling that samples more densely during fast scene
  changes.
- Showing intermediate frame-level observations and the transcript in
  the frontend, not just the final caption.
- Supporting longer videos by chunking and summarizing chunk-by-chunk.
- Returning the top few matches instead of only the single best one.
- Using a larger Whisper model for better transcription accuracy when
  GPU headroom allows.

## 29. Simple Explanation for a Viva/Interview

"I use OpenCV to sample representative video frames and a pretrained
Qwen vision-language model to understand the visual content. I
separately convert the video's speech to text using a pretrained
Whisper model (specifically faster-whisper, a lighter-weight
implementation of the same Whisper weights). I then give both the
visual observations and the transcript to Qwen so it can generate a
short, 2-3 sentence context-aware caption that reflects what it saw
*and* what was said, in chronological order.
I convert that caption into an embedding using a small pretrained
sentence-embedding model, and store it with the local video path in a
simple JSON file — no database server needed. When a user searches in
natural language, I embed their query the same way and use cosine
similarity to find the closest stored caption, then play back the
matching video in the browser."

## 30. Folder/File Structure

```
project/
│
├── context_aware_video.ipynb   # experimentation notebook (now includes audio + search)
├── main.py                     # FastAPI backend (captioning + search pipeline)
├── requirements.txt            # Python dependencies
├── index.html                  # one-page frontend (captioning + search sections)
├── PROJECT_GUIDE.md            # this file
├── video_database.json         # created automatically - the local "database"
└── videos/                     # created automatically - stored video files
```

## 31. What Each File Does

- **context_aware_video.ipynb** — step-by-step walkthrough: checking
  Ollama, sampling frames, transcribing audio, calling Qwen, embedding
  captions, storing videos, and testing semantic search.
- **main.py** — the same pipeline wrapped in a FastAPI app with two
  endpoints: `POST /caption` and `POST /search`, plus static file
  serving for the frontend and the `videos/` folder.
- **requirements.txt** — the Python packages needed.
- **index.html** — one page with two sections: video captioning, and
  semantic search with a video player for the result.
- **video_database.json / videos/** — created automatically the first
  time a video is captioned; not something you edit by hand.
- **PROJECT_GUIDE.md** — this document.

## 32. Installation Steps (Windows + Miniconda)

```
conda create -n videocaption python=3.10 -y
conda activate videocaption
pip install -r requirements.txt
```

Also install **ffmpeg** (faster-whisper needs it to read audio from
video files) — on Windows, the easiest way is:

```
conda install -c conda-forge ffmpeg
```

## 33. Ollama Setup

1. Check Ollama is installed: `ollama --version`
2. Check which models are already pulled: `ollama list`
3. Pull the vision-language model: `ollama pull qwen3-vl:8b`
4. Quick test:
   ```
   ollama run qwen3-vl:8b "describe this image" ./test_image.jpg
   ```
   Ollama's background server (`http://localhost:11434`) starts
   automatically once Ollama is installed and running on Windows.

## 34. How to Run the Notebook

1. Activate the conda environment above.
2. Launch Jupyter Lab: `jupyter lab`
3. Open `context_aware_video.ipynb`.
4. Update the video/image paths near the top of each relevant cell to
   point at files on your machine.
5. Run the cells from top to bottom.

## 35. How to Start FastAPI

From the project folder: `uvicorn main:app --reload`

This starts the server at `http://127.0.0.1:8000` and automatically
creates `videos/` and `video_database.json` on first use.

## 36. How to Use the Frontend

Open `http://127.0.0.1:8000` in a browser. In the top section, choose
a video and click **Analyze Video** to get its caption (this also
stores the video for search). In the bottom section, type a
description of a video you're looking for and click **Search** — a
matching video will play if one is found above the similarity
threshold.

## 37. API Request/Response Format

**`POST /caption`** — `multipart/form-data` with a `video` file field.
```json
{ "caption": "...", "frames_analyzed": 8 }
```

**`POST /search`** — JSON body:
```json
{ "query": "find the video where someone is cooking" }
```
Found:
```json
{
  "found": true,
  "video_id": "video_001",
  "video_path": "/videos/video_001.mp4",
  "caption": "...",
  "similarity": 0.82
}
```
Not found:
```json
{ "found": false, "message": "No matching video found in the database." }
```

## 38. Example Input/Output

**Input:** a video of someone in a kitchen, saying "I am preparing
vegetables for dinner" while cutting vegetables.

**Visual observations:**
```
Frame 1: Person standing in a kitchen.
Frame 2: Person holding vegetables.
Frame 3: Person cutting vegetables.
```

**Audio transcript:** `"I am preparing vegetables for dinner."`

**Final caption:**
```
"The person enters the kitchen and picks up vegetables from the counter.
They then begin cutting the vegetables, mentioning that they are
preparing them for dinner."
```

**Later, searching:** `"a video of someone cooking"` → returns this
video with a similarity around 0.7-0.8 and plays it back.

## Architecture Diagram

```
VIDEO
 │
 ├── Visual  -> OpenCV -> Qwen-VL -> Visual observations
 └── Audio   -> faster-whisper -> Transcript
                    │
                    ▼
       Visual observations + Transcript
                    │
                    ▼
            Qwen context reasoning
                    │
                    ▼
              Final caption
                    │
                    ▼
             Sentence embedding
                    │
                    ▼
           Local video database
```
