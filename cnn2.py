import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt 

# Load the dataset
# (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Preprocess the images
# train_images = train_images.reshape((len(train_images), 784))
# test_images = test_images.reshape((len(test_images), 784))

# x_train = np.reshape(train_images, (train_images.shape[0], 784, 784))
# x_test = np.reshape(test_images, (test_images.shape[0], 784, 784))

# original_image_shape = train_images.shape[1:]

# train_images = train_images.reshape((len(train_images), *original_image_shape))
# test_images = test_images.reshape((len(test_images), *original_image_shape))


# print(train_images.shape)
# train_images = train_images.astype("float32") / 255.0
# test_images = test_images.astype("float32") / 255.0
# print(train_images.shape)
# # Add Gaussian noise to the images
# train_images_noisy = train_images + 0.5 * tf.random.normal(shape=train_images.shape)
# test_images_noisy = test_images + 0.5 * tf.random.normal(shape=test_images.shape)




# Define the CNN architecture
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
    ]
)

model.summary()
# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy")

# Train the model
print("herrrrr")
model.fit(train_images_noisy, train_images, epochs=1, batch_size=128, validation_split=0.1)
print("zall")

# Evaluate the model
score = model.evaluate(test_images_noisy, test_images, verbose=0)
print("Test loss:", score)

# Apply the model to denoise an image
import cv2
import numpy as np

# Load an image
img = cv2.imread('girl.jpg', cv2.IMREAD_GRAYSCALE)

# img = cv2.imread("path/to/image.jpg")

# if img is None or img.size == 0:
#     print("Error: Failed to load the image or the image is empty.")
# else:
#     img = cv2.resize(img, (28, 28))
#     # Rest of your code...

# Preprocess the image
print("scsccssc")
img = cv2.resize(img, (28, 28))
img = img.astype("float32") / 255.0
print("x")

# img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)
print("xzz")

# Denoise the image
denoised_img = model.predict(img)
print("xaaaaaaaaaaaaa")

# Resize denoised image to original image size
denoised_img = cv2.resize(denoised_img[0], (img.shape[1], img.shape[2]))

# Save the denoised image
cv2.imwrite('denoised_image.png', denoised_img[0] * 255.0)

image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Plot the image
plt.imshow(denoised_img)
plt.axis('off')  # Disable axis
plt.show()