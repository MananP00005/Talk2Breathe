import html
import json
import re
from collections import Counter

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from waste_vector import retriever

model = OllamaLLM(model="llama3.2")

# Real-world recycling bin colors aren't standardized (they differ by city
# and country), so the model can't reliably know one. Instead we teach a
# single consistent 5-category system every time.
CATEGORY_INFO = {
    "recycle": {"emoji": "♻️", "color": "#1e88e5", "name": "Recycling Bin"},
    "compost": {"emoji": "🌱", "color": "#43a047", "name": "Compost Bin"},
    "landfill": {"emoji": "🗑️", "color": "#616161", "name": "Landfill / Trash"},
    "hazardous": {"emoji": "⚠️", "color": "#e53935", "name": "Hazardous Waste"},
    "ewaste": {"emoji": "🔌", "color": "#8e24aa", "name": "E-Waste Bin"},
}
FALLBACK_CATEGORY = "landfill"

# Only asks the model to classify ONE already-known item at a time, and never
# asks it to restate the item's name. Earlier, a single call classifying the
# whole detected list let the model substitute example items from the
# knowledge-base context (e.g. turning 7 detected cans into "glass jar",
# "phone charger", etc., copied straight out of the reference guidelines).
# Keeping "item" entirely out of the model's hands removes that failure mode.
template = """
You are Breathe, a friendly guide helping children aged 7-13 learn how to sort
waste and recycle correctly.

Rules you must always follow:
- Use simple, clear language a 7-year-old can understand
- Be encouraging and positive, never scary or preachy
- "category" must be EXACTLY one of: recycle, compost, landfill, hazardous, ewaste
- "explanation" is a detailed but kid-friendly waste management explanation for
  this specific item: 3-5 sentences covering (1) exactly how to prepare it
  (rinse, flatten, remove parts, etc.), (2) why it belongs in that category,
  and (3) one relevant safety or environmental note if applicable.

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"category": "recycle|compost|landfill|hazardous|ewaste", "explanation": "3-5 sentence detailed explanation"}}

Reference disposal guidelines (background only, do not copy item names from here): {docs}
Classify this exact item, nothing else: {item}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def _parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _classify_item(label: str) -> dict:
    docs = retriever.invoke(label)
    raw = chain.invoke({"docs": docs, "item": label})
    try:
        parsed = _parse_json_object(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return {
        "item": label,  # always our own known label, never the model's
        "category": parsed.get("category", FALLBACK_CATEGORY),
        "explanation": parsed.get("explanation", ""),
    }


def _render_cards(classified: list) -> str:
    cards = []
    for entry in classified:
        category = CATEGORY_INFO.get(entry["category"], CATEGORY_INFO[FALLBACK_CATEGORY])
        item = html.escape(f"{entry['count']}x {entry['item']}" if entry["count"] > 1 else entry["item"])
        explanation = html.escape(entry["explanation"])
        cards.append(
            '<div class="waste-card" style="padding:12px 16px;margin:8px 0;border-radius:14px;'
            f'background:{category["color"]}1a;border-left:6px solid {category["color"]};">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
            f'<span style="font-size:28px;">{category["emoji"]}</span>'
            f'<div><b>{item}</b> &rarr; <b style="color:{category["color"]}">{category["name"]}</b></div>'
            "</div>"
            f'<div class="waste-card-explain" style="font-size:14px;line-height:1.5;">{explanation}</div>'
            "</div>"
        )
    return "".join(cards)


def _render_speech(classified: list) -> str:
    parts = []
    for entry in classified:
        category = CATEGORY_INFO.get(entry["category"], CATEGORY_INFO[FALLBACK_CATEGORY])
        parts.append(f"{entry['item']} goes in the {category['name']}. {entry['explanation']}")
    return " ".join(parts)


def get_waste_instructions(detections: list) -> tuple[str, str]:
    """Returns (html_for_chat, plain_text_for_speech)."""
    if not detections:
        message = "I couldn't spot anything in that picture — want to try another one?"
        return message, message

    counts = Counter(d["label"] for d in detections)
    classified = [
        {**_classify_item(label), "count": count} for label, count in counts.items()
    ]

    return _render_cards(classified), _render_speech(classified)
