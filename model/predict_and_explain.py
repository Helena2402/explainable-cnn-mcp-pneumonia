import numpy as np
import tensorflow as tf

from model.gradcam import compute_gradcam

import cv2
import numpy as np

def visualize_gradcam(heatmap, original_image):
    # resize heatmap
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # normalize to 0–255
    heatmap = np.uint8(255 * heatmap)

    # apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # convert original to RGB if needed
    if original_image.shape[-1] == 1:
        original_image = np.repeat(original_image, 3, axis=-1)

    # scale original image
    original_image = np.uint8(255 * original_image)

    # overlay
    superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    return superimposed


class PneumoniaModelService:
    """
    Wraps CNN prediction + Grad-CAM explanation
    as a reusable service component.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def predict_and_explain(self, image: np.ndarray) -> dict:
        """
        Parameters
        ----------
        image : np.ndarray
            Shape (28, 28, 1), values in [0, 1]

        Returns
        -------
        dict with keys:
            - prediction
            - probability
            - heatmap
        """

        # Enforce correct shape
        if image.shape != (1, 28, 28, 1):
            raise ValueError(f"predict_and_explain expected (1,28,28,1), got {image.shape}")

        image_batch = image

        # Prediction
        prob = float(self.model.predict(image_batch)[0][0])
        prediction = int(prob >= 0.5)

        # Grad-CAM
        heatmap = compute_gradcam(
            model=self.model,
            image=image_batch,
            conv_layer_name="conv_block_last",
        )

        image_np = image_batch[0]  # remove batch

        heatmap = visualize_gradcam(heatmap, image_np)

        return {
            "prediction": prediction,
            "probability": prob,
            "heatmap": heatmap,
        }
