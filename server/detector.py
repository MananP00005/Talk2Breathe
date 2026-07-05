# || SWAP POINT ||
# This file is a placeholder for the real detector: your friend's YOLO model
# (trained the same way as CLEAR-AI's server/detector.py — cv2.imdecode +
# ultralytics YOLO().predict()). Until that model/weights are available,
# Gemini's vision API plays the same role: look at the image, find the
# waste-relevant object(s), report a label + confidence + bounding box.
#
# The contract below is written to match YOLO's output shape exactly:
#   detect_image(image_bytes: bytes) -> list[{"label": str, "confidence": float, "box": [x1, y1, x2, y2]}]
#
# To swap in the real model later: replace the body of detect_image() with
# YOLO inference (see CLEAR-AI's detector.py). Nothing in main.py or waste.py
# needs to change — they only depend on this return shape.
# || End of note ||

import os
import json
import re
from io import BytesIO

from PIL import Image
from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-2.5-flash"

_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to server/.env (see .env.example)."
    )

client = genai.Client(api_key=_api_key)

DETECTION_PROMPT = """You are a waste-item detector. Look at this image and identify every
distinct physical object that is relevant to waste disposal or recycling
(e.g. bottle, can, food container, wrapper, battery, cigarette butt, electronic device, paper, box).

Respond with ONLY a JSON array, no other text, no markdown fences. Each entry:
{"label": "short lowercase name", "confidence": 0.0-1.0, "box_2d": [ymin, xmin, ymax, xmax]}

box_2d values are normalized to a 0-1000 scale relative to the image height/width.
If nothing relevant is found, respond with an empty array: []"""


def _parse_json_array(text: str) -> list:
    # Gemini occasionally wraps JSON in ```json fences despite instructions
    # not to. Strip fences before parsing.
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _box_2d_to_xyxy(box_2d: list, width: int, height: int) -> list:
    # Gemini returns [ymin, xmin, ymax, xmax] normalized to 0-1000.
    # Convert to pixel-space [x1, y1, x2, y2] to match YOLO's box format.
    ymin, xmin, ymax, xmax = box_2d
    x1 = round((xmin / 1000) * width)
    y1 = round((ymin / 1000) * height)
    x2 = round((xmax / 1000) * width)
    y2 = round((ymax / 1000) * height)
    return [x1, y1, x2, y2]


def detect_image(image_bytes: bytes) -> list:
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                DETECTION_PROMPT,
            ],
        )
    except Exception as e:
        # Network hiccups, rate limits, transient API errors, etc. shouldn't
        # crash the whole chat turn -- treat it the same as "nothing found".
        print(f"[detector] Gemini API call failed: {e}")
        return []

    try:
        raw_detections = _parse_json_array(response.text)
    except (json.JSONDecodeError, TypeError):
        return []

    detections = []
    for item in raw_detections:
        label = item.get("label")
        box_2d = item.get("box_2d")
        if not label or not box_2d:
            continue
        detections.append({
            "label": label,
            "confidence": round(float(item.get("confidence", 0.5)), 2),
            "box": _box_2d_to_xyxy(box_2d, width, height),
        })

    return detections
