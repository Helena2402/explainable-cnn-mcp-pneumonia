import numpy as np
import tensorflow as tf

from model.gradcam import compute_gradcam


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

        return {
            "prediction": prediction,
            "probability": prob,
            "heatmap": heatmap.tolist(),
        }
