import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from model.cnn_functional import build_functional_cnn


from model.gradcam import compute_gradcam



# -------------------------
# Load trained model
# -------------------------
# Load trained Sequential model
seq_model = tf.keras.models.load_model("model/pneumonia_cnn.keras")

# Build functional model
model = build_functional_cnn()

# Transfer weights
model.set_weights(seq_model.get_weights())


# -------------------------
# Load PneumoniaMNIST test set
# -------------------------
ds_test = tfds.load(
    "pneumonia_mnist",
    split="test",
    as_supervised=True
)

# Take one sample
image, label = next(iter(ds_test.take(1)))

# Add batch dimension
image_batch = tf.expand_dims(image, axis=0)

print("True label:", int(label.numpy()))

# -------------------------
# Model prediction
# -------------------------
pred = model.predict(image_batch)
pred_prob = float(pred[0][0])
pred_class = int(pred_prob >= 0.5)

print(f"Predicted probability: {pred_prob:.4f}")
print(f"Predicted class: {pred_class}")

# -------------------------
# Grad-CAM
# -------------------------
# Force model to build its computation graph
_ = model(image_batch)


heatmap = compute_gradcam(
    model=model,
    image=image_batch,
    conv_layer_name="conv_block_last",
)


print("Heatmap shape:", heatmap.shape)

# -------------------------
# Resize heatmap to image size
# -------------------------
heatmap_resized = tf.image.resize(
    heatmap[..., tf.newaxis],
    (28, 28)
).numpy().squeeze()


# -------------------------
# Plot original + Grad-CAM
# -------------------------
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Original image")
plt.imshow(image.numpy().squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(image.numpy().squeeze(), cmap="gray")
plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.show()