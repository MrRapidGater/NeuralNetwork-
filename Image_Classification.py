import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define a CNN model

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


# Train the model for  epochs

history = model.fit(
    train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plot training history
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")
plt.show()


# Save the trained model
model.save("cifar10_cnn_model")

# Load the saved model
model = tf.keras.models.load_model("cifar10_cnn_model")

# Load an image

# IMAGE MUST BE OF CLASS NAMES LISTED BELOW AND SIZE 32x32
image_path = r"C:\Users\Ali Kumayl\Desktop\Air-New-Zealand-Boeing-747-400.webp"  # <<----------------------- PUT IMAGE PATH HERE
image = Image.open(image_path)
image = image.resize((32, 32))  # Resize the image to match the input shape of the model
image = np.array(image) / 255.0  # Normalize the pixel values

# Add batch dimension and predict the class probabilities
prediction = model.predict(np.expand_dims(image, axis=0))

# Define the class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Get the predicted class label
predicted_label = np.argmax(prediction)

# Print the predicted class name
predicted_class_name = class_names[predicted_label]
print(f"The predicted class is: {predicted_class_name}")
