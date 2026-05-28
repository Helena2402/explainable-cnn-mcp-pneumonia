# server.py

import tensorflow as tf
import numpy as np
from fastmcp import FastMCP

from model.cnn_functional import build_functional_cnn
from model.predict_and_explain import PneumoniaModelService

# -------------------------
# Load model
# -------------------------
seq_model = tf.keras.models.load_model("model/pneumonia_cnn.keras")
model = build_functional_cnn()
model.set_weights(seq_model.get_weights())

service = PneumoniaModelService(model)

# -------------------------
# MCP server
# -------------------------
mcp = FastMCP("Pneumonia CNN MCP Server")


@mcp.tool()
def predict_pneumonia(image: list) -> dict:
    """
    Predict pneumonia and provide Grad-CAM explanation.

    image: nested list with shape (28, 28, 1)
    """
    image_np = np.array(image, dtype=np.float32)

    return service.predict_and_explain(image_np)


if __name__ == "__main__":
    mcp.run("streamable-http")