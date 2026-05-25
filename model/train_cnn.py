import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


from .cnn_model import build_cnn

def load_data():
    """
    Loads the PneumoniaMNIST dataset via TensorFlow Datasets.
    """

    ds_train, ds_val, ds_test = tfds.load(
        "pneumonia_mnist",
        split=["train", "val", "test"],
        as_supervised=True
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds_train = ds_train.map(preprocess).batch(64)
    ds_val = ds_val.map(preprocess).batch(64)
    ds_test = ds_test.map(preprocess).batch(64)

    return ds_train, ds_val, ds_test

def main():
    ds_train, ds_val, ds_test = load_data()

    model = build_cnn(input_shape=(28, 28, 1))

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=10
    )

    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save trained model
    model.save("model/pneumonia_cnn.keras")
    print("Model saved to model/pneumonia_cnn.keras")


if __name__ == "__main__":
    main()