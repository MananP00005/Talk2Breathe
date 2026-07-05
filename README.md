# Talk2Breathe-AI

A child-centered AI chatbot: a smoking-prevention text chatbot, plus an image-upload waste-sorting feature. Single-process Gradio app, local RAG via Ollama, Gemini Vision as a stand-in for a not-yet-available trained YOLO detector.

The Vue-based version of this project has moved to the standalone sibling project `Talk2Breathe-Vue`.

---

## Prerequisites

- Python 3.10+
- Ollama

---

## Setup

### 1. Install Ollama and pull the required models

```bash
brew install ollama          # or: curl -fsSL https://ollama.com/install.sh | sh
brew services start ollama
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure the Gemini API key (waste-detection stand-in)

Image uploads are handled by `server/detector.py`, which currently calls the
Gemini API to identify the item in the photo instead of a trained model.
This is a temporary stand-in — see "Swapping in the trained YOLO model" below.

```bash
cp server/.env.example server/.env
# then edit server/.env and paste in a key from https://aistudio.google.com/apikey
```

---

## Run the app

```bash
cd server
source ../venv/bin/activate
python app.py
```

Open `http://localhost:7860`. Chat with Breathe, or upload a photo of an
item to get sorting instructions — click "🔊 Listen" on any message to hear
it read aloud (uses the browser's built-in text-to-speech, no server audio
generation involved).

---

## Swapping in the trained YOLO model

Image detection currently goes through Gemini's vision API
(`server/detector.py`) as a placeholder, because the real model is a
separately trained YOLO model (see
[CLEAR-AI](https://github.com/Fennerii/CLEAR-AI) for the reference
implementation and training script) that isn't available in this repo yet.

`detector.py` exposes one function with a fixed contract:

```python
detect_image(image_bytes: bytes) -> list[{"label": str, "confidence": float, "box": [x1, y1, x2, y2]}]
```

To swap in the trained model once you have `best.pt`:

1. Add `ultralytics` and `opencv-python` to `requirements.txt` (remove
   `google-genai` if no longer needed).
2. Replace the body of `detect_image()` in `server/detector.py` with YOLO
   inference: decode `image_bytes` with `cv2.imdecode`, run
   `YOLO("best.pt").predict(...)`, and map each detected box to the same
   `{"label", "confidence", "box"}` shape.
3. Nothing else needs to change — `app.py` and `waste.py` only depend on
   that return shape.

**Note on licensing:** CLEAR-AI's repository has no LICENSE file, meaning
its code and trained weights are all-rights-reserved by default. Get
explicit permission before reusing `best.pt` or copying its code directly.

---

## Waste-disposal knowledge base

Drop recycling/disposal guideline PDFs, DOCX, or HTML files into
`server/docs/waste/`. On first run with no files there, `waste_vector.py`
falls back to a small built-in seed document so the feature still works.
Once you add real guideline files, delete `server/chroma_db_waste/` so it
rebuilds from them.
