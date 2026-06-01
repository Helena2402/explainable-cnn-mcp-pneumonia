import streamlit as st
import asyncio
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from langchain_core.messages import AIMessage

import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf

# --------------------------------------------------
# Project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from agent.langgraph_agent import build_agent


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_tensor" not in st.session_state:
    st.session_state.image_tensor = None

if "heatmap" not in st.session_state:
    st.session_state.heatmap = None

if "graph" not in st.session_state:
    st.session_state.graph = None


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L").resize((28, 28))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img[..., None][None, ...]  # (1, 28, 28, 1)
    return img


def render_message(msg):
    if isinstance(msg, dict):
        role, content = msg["role"], msg["content"]
    elif isinstance(msg, AIMessage):
        role, content = "assistant", msg.content
    else:
        role, content = "assistant", str(msg)

    with st.chat_message(role):
        st.write(content)


# --------------------------------------------------
# Init agent
# --------------------------------------------------

async def init_agent():
    return await build_agent()

if st.session_state.graph is None:
    st.session_state.graph = asyncio.run(init_agent())


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("Pneumonia Detection – Explainable CNN")

# Fixed two-column layout
col_img, col_heat = st.columns(2)

# -------- Left: X-ray upload --------
with col_img:
    uploaded = st.file_uploader(
        "Upload chest X‑ray (28×28, one image)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded X‑ray (28×28)", width=200)

        st.session_state.image_tensor = preprocess_image(img)
        st.session_state.heatmap = None  # reset old heatmap

        st.write("DEBUG image tensor:", st.session_state.image_tensor.shape)

# -------- Chat input + agent call --------
user_input = st.chat_input("Ask something…", key="chat_input")

if user_input:
    # Append user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Call LangGraph agent ONCE
    result = asyncio.run(
        st.session_state.graph.ainvoke(
            {
                "messages": st.session_state.messages,
                "image": (
                    st.session_state.image_tensor.tolist()
                    if st.session_state.image_tensor is not None
                    else None
                ),
            }
        )
    )

    # Append new assistant messages (no duplication)
    st.session_state.messages.extend(
        result["messages"][len(st.session_state.messages):]
    )

    # Store heatmap if present
    if "heatmap" in result:
        heatmap = np.array(result["heatmap"])

        # Ensure 2D
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze()

        # Resize 11×11 → 28×28 (CRITICAL FIX)
        if heatmap.shape != (28, 28):
            heatmap = tf.image.resize(
                heatmap[..., tf.newaxis],
                (28, 28)
            ).numpy().squeeze()

        st.session_state.heatmap = heatmap


# -------- Right: Grad‑CAM rendering (AFTER state update) --------
with col_heat:


    if st.session_state.heatmap is not None and st.session_state.image_tensor is not None:
        st.markdown("### Grad‑CAM Heatmap")
        image = st.session_state.image_tensor[0, :, :, 0]
        heatmap = st.session_state.heatmap

        # Normalise heatmap
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Matplotlib overlay (same as working example)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image, cmap="gray")
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        st.image(buf, caption="Grad‑CAM overlay (28×28)")


# -------- Chat history --------
st.divider()

for m in st.session_state.messages:
    render_message(m)