import numpy as np
import tensorflow as tf
from fastmcp import FastMCP

from model.cnn_functional import build_functional_cnn
from model.predict_and_explain import PneumoniaModelService


# --------------------------------------------------
# Load model
# --------------------------------------------------

seq_model = tf.keras.models.load_model("model/pneumonia_cnn.keras")
model = build_functional_cnn()
model.set_weights(seq_model.get_weights())

service = PneumoniaModelService(model)

# --------------------------------------------------
# MCP server
# --------------------------------------------------

mcp = FastMCP("Pneumonia CNN MCP Server")


@mcp.tool()
def predict_pneumonia(image: list) -> dict:
    if image is None or len(image) == 0:
        raise ValueError("No image provided to predict_pneumonia")

    image_np = np.array(image, dtype=np.float32)

    # Fix possible channel-first layout after serialisation
    if image_np.ndim == 4 and image_np.shape[1] == 1 and image_np.shape[-1] != 1:
        image_np = np.transpose(image_np, (0, 2, 3, 1))

    if image_np.shape != (1, 28, 28, 1):
        raise ValueError(f"Invalid image shape after fix: {image_np.shape}")

    result = service.predict_and_explain(image_np)

    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
    }


@mcp.tool()
def explain_pneumonia(image: list) -> dict:
    if image is None or len(image) == 0:
        raise ValueError("No image provided to explain_pneumonia")

    image_np = np.array(image, dtype=np.float32)

    if image_np.ndim == 4 and image_np.shape[1] == 1 and image_np.shape[-1] != 1:
        image_np = np.transpose(image_np, (0, 2, 3, 1))

    if image_np.shape != (1, 28, 28, 1):
        raise ValueError(f"Invalid image shape after fix: {image_np.shape}")

    result = service.predict_and_explain(image_np)

    return {
        "heatmap": result["heatmap"]
    }


if __name__ == "__main__":
    mcp.run("streamable-http")
