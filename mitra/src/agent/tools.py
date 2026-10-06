"""Robot capabilities as Strands tools (DESIGN §1.3).

Tools are built by a factory bound to live robot/TTS instances so tests can
inject fakes. Two invariants from DESIGN §1.4: ``nod`` also fires
deterministically on wake, and every reply is routed through the validator and
spoken by the orchestrator — the model has no speech tool, so it cannot speak
unvalidated text or speak a reply twice.
"""

from __future__ import annotations

import io

import numpy as np

try:
    from strands import tool
except ImportError:  # plain-loop fallback mode (FR-6.2): identical interface
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

END_SESSION_SENTINEL = "session_end"


def _encode_jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image

    # reachy-mini serves frames in OpenCV BGR order; the VLM needs RGB
    buf = io.BytesIO()
    Image.fromarray(frame[..., ::-1]).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def build_tools(robot, tts) -> list:
    """The agent tools over a robot + TTS pair (real or fake).

    There is deliberately no speech tool. The orchestrator speaks every final
    reply after validation (DESIGN §1.4); a model-callable ``speak_sanskrit``
    played unvalidated text and, when the model used it, the same sentence
    was spoken twice. ``tts`` stays in the signature for callers.
    """

    @tool
    def capture_image() -> dict:
        """Capture one frame from Reachy Mini's camera. Use when the user shows
        an object or asks what something is."""
        frame = robot.camera_read()
        return {"format": "jpeg", "source": {"bytes": _encode_jpeg(frame)}}

    @tool
    def nod() -> str:
        """Briefly nod the robot's head to acknowledge the user."""
        robot.nod()
        return "ok"

    @tool
    def end_session() -> str:
        """End the conversation and return to wake-word listening. Call when
        the user says goodbye."""
        return END_SESSION_SENTINEL

    return [capture_image, nod, end_session]
