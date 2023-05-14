import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense,Reshape, Input
from keras.callbacks import ModelCheckpoint
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from skimage.io import imread , imsave
# from skimage.transform import resize
from denoise_img import non_local_means_denoise

# Load the noisy image
img = cv2.imread('girl.jpg')

print('Target data type:', img.dtype)
print('Target shape type:', img.shape)


# img = img_as_float(img)
# Resize the image
img = cv2.resize(img, (92, 92),interpolation=cv2.INTER_CUBIC)
# Ensure the image has 3 channels
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Expand dimensions to match model input shape
img = np.expand_dims(img, axis=0)
print('Resized image shape:', img.shape)
# Denoise the image using Non-Local Means algorithm
# sigma_est = np.mean(estimate_sigma(img, multichannel=True))
# h = 0.2
# sigma = 0.5
# denoised_img = denoise_nl_means(img, h=1.15 * sigma, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)


# Define the CNN model
# model = Sequential()
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(92, 92, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Flatten())

# model.add(Dense(256))  # Reshape the tensor to (None, 67712)
# # model.add(Reshape((92, 92, 32)))  # Reshape the tensor to (None, 92, 92, 32)
# model.add(Reshape((8, 8, 4)))

# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model = Sequential(
    [
        Input(shape=(92, 92, 3)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2), padding="same"),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2), padding="same"),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        UpSampling2D((2, 2)),
        Conv2D(3, (3, 3), activation="sigmoid", padding="same"),
    ]
)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(img, img, batch_size=1, epochs=50, callbacks=[checkpoint],)

# validation_split=0.2,
print("model done")
model.save('best_model.h5')
# Load the best model
model.load_weights('best_model.h5')
print("yep")

# Resize the image to the expected input shape of the model
img = cv2.resize(img, (92, 92),interpolation=cv2.INTER_CUBIC)

# Add an extra dimension to represent the batch size
img = np.expand_dims(img, axis=0)
print("gud")

# Predict the denoised image using the model
denoised_image = model.predict(img)
print("great")

# Save the denoised image
imsave('denoised_image.jpg', denoised_image[0])
print('Input data shape:', denoised_image.shape)
print('Input data type:', denoised_image.dtype)






