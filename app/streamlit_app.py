import streamlit as st
import asyncio
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from langchain_core.messages import AIMessage

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
    img = img[..., None][None, ...]
    return img


def render_message(msg):
    if isinstance(msg, dict):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

    elif hasattr(msg, "content"):
        role = "assistant"
        content = msg.content

    else:
        return

    # ✅ Style decision messages
    if isinstance(content, str) and content.startswith("DECISION:"):
        label = content.split(":")[1].strip()

        with st.chat_message("assistant"):
            st.markdown(
                f"<span style='color: gray;'>⚙️ Decision → {label}</span>",
                unsafe_allow_html=True
            )
        return

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

col_img, col_heat = st.columns(2)

# -------- Image upload --------
with col_img:
    uploaded = st.file_uploader(
        "Upload chest X-ray",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)

        st.session_state.image_tensor = preprocess_image(img)


# -------- Chat --------
user_input = st.chat_input("Ask something…")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    prev_len = len(st.session_state.messages)

    state = {
        "messages": st.session_state.messages,
        "image": st.session_state.image_tensor,
        "heatmap": st.session_state.heatmap,
    }

    result = st.session_state.graph.invoke(state)

    new_messages = result["messages"][prev_len:]

    for m in new_messages:
        if isinstance(m, AIMessage) and m.content.strip():
            st.session_state.messages.append(m)

    # ✅ update heatmap
    if "heatmap" in result:
        st.session_state.heatmap = result["heatmap"]
# -------- Heatmap (FINAL CLEAN VERSION) --------
with col_heat:
    if (
        st.session_state.heatmap is not None
        and st.session_state.image_tensor is not None
    ):
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from io import BytesIO
        import cv2

        st.markdown("### Grad‑CAM Heatmap")

        # ✅ Original image
        image = st.session_state.image_tensor[0, :, :, 0]

        # ✅ Heatmap
        heatmap = np.array(st.session_state.heatmap)

        # ✅ Fix wrong shape (28,28,28 → 28,28)
        if heatmap.ndim == 3:
            heatmap = np.mean(heatmap, axis=-1)

        heatmap = heatmap.squeeze()

        # ✅ Resize to smoother resolution (KEY for good look)
        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            (128, 128),
            method="bilinear"
        ).numpy().squeeze()

        image_big = tf.image.resize(
            image[..., np.newaxis],
            (128, 128)
        ).numpy().squeeze()

        # ✅ Smooth heatmap (makes it look like your old version)
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)

        # ✅ Normalize
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # ✅ Overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_big, cmap="gray")
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        st.image(buf, caption="Grad‑CAM overlay")
# -------- Chat history --------
st.divider()

for m in st.session_state.messages:
    render_message(m)
