import html
import random

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from chatbot import get_response
from detector import detect_image
from waste import get_waste_instructions

GREETING = (
    "Hi! I'm Breathe 🌱 Ask me a question, or upload a photo of an item "
    "and I'll help you sort it into the right bin."
)

THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Baloo 2"), "sans-serif"],
)

# A handful of simple, recognizable shapes (leaf, star, bubble, cloud) drawn
# as plain inline SVG paths -- no external icon fetch, no image files.
SHAPE_PATHS = {
    "leaf": '<path d="M12 2C7 2 3 6 3 12c0 5 4 9 9 10 1-5 5-9 10-10-1-6-5-10-10-10z" fill="{c}"/>',
    "star": '<path d="M12 2l2.9 6.9 7.1.6-5.4 4.8 1.6 7.2L12 17.8 5.8 21.5l1.6-7.2L2 9.5l7.1-.6z" fill="{c}"/>',
    "bubble": '<circle cx="12" cy="12" r="10" fill="{c}" opacity="0.85"/>',
    "cloud": '<path d="M6 17a4 4 0 010-8 5 5 0 019.6-1.5A4.5 4.5 0 0119 17H6z" fill="{c}"/>',
}
SHAPE_COLORS = ["#39b54a", "#1e88e5", "#fbb03b", "#8e24aa", "#ff6b6b", "#26c6da"]


def build_floating_shapes(count: int = 16) -> str:
    """Builds the decorative animated-SVG background, once per app launch.

    Random per shape (position/size/speed), baked into static HTML server-side
    since Gradio's gr.HTML content isn't re-randomized client-side like the
    Vue version's JS Math.random() was -- this gets the same organic effect.
    """
    kinds = list(SHAPE_PATHS.keys())
    shapes = []
    for _ in range(count):
        kind = random.choice(kinds)
        color = random.choice(SHAPE_COLORS)
        size = random.randint(22, 46)
        top = random.randint(0, 92)
        left = random.randint(0, 94)
        duration = round(random.uniform(9, 22), 1)
        delay = round(random.uniform(-18, 0), 1)
        inner = SHAPE_PATHS[kind].format(c=color)
        shapes.append(
            f'<svg class="floaty" viewBox="0 0 24 24" style="'
            f"top:{top}%;left:{left}%;width:{size}px;height:{size}px;"
            f'animation-duration:{duration}s;animation-delay:{delay}s;">{inner}</svg>'
        )
    return f'<div class="floaty-bg">{"".join(shapes)}</div>'


def with_listen_button(display_html: str, speech_text: str) -> str:
    """Appends a small 'Listen' button to a bot message.

    Uses the browser's own speechSynthesis (Web Speech API) instead of
    server-generated audio (gTTS) -- no network round trip, no waveform
    player UI, works instantly, and because the text is baked into the
    button's own data attribute, it also lets you replay ANY past message,
    not just the latest one.
    """
    escaped = html.escape(speech_text, quote=True)
    button = (
        f'<button class="listen-btn" data-speech="{escaped}" '
        'onclick="window.t2bSpeak(this)">🔊 Listen</button>'
    )
    return f"{display_html}{button}"


# Defined once in <head> (real page load, not innerHTML-inserted -- so it's
# guaranteed to actually run, unlike a <script> tag placed inside a chat
# message, which browsers do not execute).
HEAD_JS = """
<script>
window.t2bSpeak = function(btn) {
    const text = btn.getAttribute('data-speech');
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.95;
    utter.pitch = 1.05;
    window.speechSynthesis.speak(utter);
};
</script>
"""

# Runs client-side right after a response is rendered. Clicking the last
# Listen button reuses the exact same speech path as a manual click, so
# there's only one code path to keep working.
AUTO_READ_JS = """
(readAloud) => {
    if (!readAloud) return;
    setTimeout(() => {
        const buttons = document.querySelectorAll('.listen-btn');
        if (buttons.length) buttons[buttons.length - 1].click();
    }, 200);
}
"""

# Real Gradio 6 compiled component classes (confirmed from the installed
# package's CSS, not guessed): .bubble-wrap, .panel.user-row/.bot-row,
# .message-bubble-border, .submit-button, .input-container for the chat;
# .checkbox-container / input[type=checkbox] for the toggle.
CSS = """
.gradio-container {
    font-size: 17px !important;
    background: linear-gradient(135deg, #eaf6ec, #e3f2fd) !important;
    position: relative;
    overflow-x: hidden;
}

.floaty-bg {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}
.floaty {
    position: absolute;
    opacity: 0.55;
    animation-name: floaty-move;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
}
@keyframes floaty-move {
    0%   { transform: translate(0, 0) rotate(0deg); }
    50%  { transform: translate(16px, -26px) rotate(12deg); }
    100% { transform: translate(0, 0) rotate(0deg); }
}

.gradio-container > .main, .gradio-container .contain {
    position: relative;
    z-index: 1;
}

#header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #9daecc;
    border-radius: 16px;
    padding: 14px 22px;
    margin-bottom: 12px;
}
#header-bar span { font-weight: 700; font-size: 20px; color: #1a1a1a; }

/* Real iOS-style toggle switch, built from the actual checkbox input */
.checkbox-container { gap: 8px !important; }
.checkbox-container .label-text { font-weight: 600; font-size: 14px; }
.checkbox-container input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 44px !important;
    height: 24px !important;
    min-width: 44px !important;
    border-radius: 999px !important;
    background: #ccc !important;
    background-image: none !important;
    position: relative !important;
    cursor: pointer;
    transition: background 0.2s ease;
}
.checkbox-container input[type="checkbox"]::before {
    content: "";
    position: absolute;
    top: 3px;
    left: 3px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.checkbox-container input[type="checkbox"]:checked {
    background: #39b54a !important;
    background-image: none !important;
}
.checkbox-container input[type="checkbox"]:checked::before {
    transform: translateX(20px);
}

.bubble-wrap { background: #f6faf6 !important; }

.panel.user-row { background: #39b54a !important; border: none !important; }
.panel.user-row .message-bubble-border {
    border: none !important;
    border-radius: 18px 18px 4px 18px !important;
}
.panel.user-row, .panel.user-row * { color: #ffffff !important; }

.panel.bot-row { background: #ffffff !important; border: 1px solid #e2e2e2 !important; }
.panel.bot-row .message-bubble-border {
    border: none !important;
    border-radius: 18px 18px 18px 4px !important;
}
.panel.bot-row { color: #1a1a1a !important; }
.waste-card-explain { color: #444; }

.listen-btn {
    display: inline-block;
    margin-top: 8px;
    border: 1px solid #cfe0d1;
    background: #f3f9f4;
    color: #1a1a1a;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    cursor: pointer;
}

.input-container {
    border-radius: 999px !important;
    border: 1px solid #cfe0d1 !important;
    background: #ffffff !important;
}
.submit-button { background: #39b54a !important; color: white !important; border-radius: 999px !important; }

/* Dark mode: Gradio adds a "dark" class to the root when the OS/browser
   prefers dark or ?__theme=dark is set. The light-mode colors above
   (white header/bubbles/buttons) are hardcoded and don't adapt on their
   own, so every surface needs an explicit dark equivalent here. */
.dark .gradio-container {
    background: linear-gradient(135deg, #142016, #101a22) !important;
}
.dark #header-bar {
    background: #1e2620;
    border-color: #33443a;
}
.dark #header-bar span { color: #f0f0f0; }
.dark .checkbox-container .label-text { color: #e0e0e0; }

.dark .bubble-wrap { background: #101610 !important; }

.dark .panel.bot-row { background: #1e2620 !important; border-color: #33443a !important; }
.dark .panel.bot-row { color: #f0f0f0 !important; }
.dark .waste-card-explain { color: #cfcfcf; }

.dark .listen-btn {
    background: #26332a;
    border-color: #3d5245;
    color: #e0e0e0;
}

.dark .input-container {
    background: #1e2620 !important;
    border-color: #33443a !important;
}
.dark .input-container textarea, .dark .input-container input {
    color: #f0f0f0 !important;
}
"""


def respond(message: dict, history: list):
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    history = history + [{"role": "user", "content": {"path": f}} for f in files]
    if text:
        history.append({"role": "user", "content": text})

    try:
        if files:
            with open(files[0], "rb") as f:
                image_bytes = f.read()
            detections = detect_image(image_bytes)
            reply_html, reply_speech = get_waste_instructions(detections)
            if detections:
                chips = "  ·  ".join(
                    f"**{d['label']}** ({round(d['confidence'] * 100)}%)" for d in detections
                )
                reply = f"{chips}\n\n{reply_html}"
            else:
                reply = reply_html
            speech_text = reply_speech
        elif text:
            reply = get_response(text)
            speech_text = reply
        else:
            reply = "Try typing a question, or upload a photo of an item!"
            speech_text = reply
    except Exception as e:
        # Ollama being unreachable, a malformed model response, etc. --
        # anything here should degrade to a friendly chat message instead of
        # crashing all bound outputs (every output shows as errored when a
        # Gradio event handler raises).
        print(f"[respond] error handling message: {e}")
        reply = "Oops, something went wrong on my end — can you try that again?"
        speech_text = reply

    history.append({"role": "assistant", "content": with_listen_button(reply, speech_text)})

    return history, gr.MultimodalTextbox(value=None, interactive=True)


with gr.Blocks(title="Talk2Breathe") as demo:
    gr.HTML(build_floating_shapes())

    with gr.Row(elem_id="header-bar"):
        gr.HTML("<span>🌬️ Talk2Breathe</span>")
        read_aloud = gr.Checkbox(label="🔊 Read aloud", value=True)

    chatbot_ui = gr.Chatbot(
        height=520,
        value=[{"role": "assistant", "content": with_listen_button(GREETING, GREETING)}],
        sanitize_html=False,  # we build the listen buttons' + item cards' HTML ourselves and escape user/model text
        layout="bubble",
    )
    msg = gr.MultimodalTextbox(
        file_types=["image"],
        file_count="single",
        placeholder="Ask Breathe a question, or upload a photo...",
        show_label=False,
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot_ui],
        outputs=[chatbot_ui, msg],
    ).then(fn=None, inputs=[read_aloud], js=AUTO_READ_JS)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD_JS)
