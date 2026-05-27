from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_cnn(input_shape=(28, 28, 1)):
    """
    Builds a simple CNN for binary pneumonia classification.

    Args:
        input_shape (tuple): Shape of input images (H, W, C)

    Returns:
        model (keras.Model): Compiled CNN model
    """

    model = Sequential(name="Pneumonia_CNN")

    # Convolutional block 1
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional block 2
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv_block_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Classification head
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    # Compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model