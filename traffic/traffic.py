import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    for category in range(NUM_CATEGORIES):
        new_path = os.path.join(data_dir, str(category))
        images.extend([cv2.resize(cv2.imread(os.path.join(new_path, file)),
                                  (IMG_WIDTH, IMG_HEIGHT),
                                  interpolation=cv2.INTER_CUBIC)
                       for file in os.listdir(new_path)])
        labels.extend([category] * len(os.listdir(new_path)))
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # Rescaling it to normalize the pixel values
        tf.keras.layers.Rescaling(
            1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Conv2D layer with 32 filters, kernel size of 3 x 3
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

        # Batch normalization improves model accuracy and stability by normalizing layer inputs,
        # which reduces internal covariate shift and speeds up training
        tf.keras.layers.BatchNormalization(),

        # Adding another layer, before maxpooling,
        # will give us both a detailed view, and the same broader overview as a 5,5 kernel would
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

        # Another batch normalization
        tf.keras.layers.BatchNormalization(),

        # MaxPooling2D layer to reduce the spatial dimensions by 50 %
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Dropout layer to prevent overfitting,
        # setting it to randomly set a % of the units to 0 when training
        tf.keras.layers.Dropout(0.25),


        # Repeating for 3. and 4. layer - but using 64 filters now
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Repeating for 5. layer - but using 128 filters now
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flatten from 3 dimensoin output to one dimension to prepare for vanilla feed-forward network
        tf.keras.layers.Flatten(),

        # Denselayer/fully-connected layer: 256 neurons
        tf.keras.layers.Dense(256, activation='relu'),

        # Another dropout to prevent overfitting
        tf.keras.layers.Dropout(0.25),

        # Output layer with softmax activation so we can get classifiation to the num_cat
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')

    ])

    # Finally compiling the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()
