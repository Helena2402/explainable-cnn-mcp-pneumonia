import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from model.cnn_functional import build_functional_cnn
from model.predict_and_explain import PneumoniaModelService

# Load trained Sequential model
seq_model = tf.keras.models.load_model("model/pneumonia_cnn.keras")

# Build functional model and load weights
model = build_functional_cnn()
model.set_weights(seq_model.get_weights())

service = PneumoniaModelService(model)

# Load one test image
ds_test = tfds.load("pneumonia_mnist", split="test", as_supervised=True)
image, label = next(iter(ds_test.take(1)))

result = service.predict_and_explain(image.numpy())

print("True label:", int(label.numpy()))
print("Prediction:", result["prediction"])
print("Probability:", result["probability"])
print("Heatmap shape:", len(result["heatmap"]), "x", len(result["heatmap"][0]))
