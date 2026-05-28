import numpy as np
import tensorflow as tf
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
    Predict pneumonia probability from a chest X-ray.
    """
    image_np = np.array(image, dtype=np.float32)

    result = service.predict_and_explain(image_np)

    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
    }


@mcp.tool()
def explain_pneumonia(image: list) -> dict:
    """
    Generate Grad-CAM explanation for a chest X-ray.
    """
    image_np = np.array(image, dtype=np.float32)

    result = service.predict_and_explain(image_np)

    return {
        "heatmap": result["heatmap"]
    }


if __name__ == "__main__":
    mcp.run("streamable-http")