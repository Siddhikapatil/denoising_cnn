import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense,Reshape, Input
from keras.callbacks import ModelCheckpoint
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from skimage.io import imread , imsave
from skimage.transform import resize
from denoise_img import non_local_means_denoise

# Load the noisy image
img = cv2.imread('girl.jpg')
# img = img_as_float(img)
print('Target data type:', img.dtype)
print('Target shape type:', img.shape)
# Denoise the image using Non-Local Means algorithm
# sigma_est = np.mean(estimate_sigma(img, multichannel=True))
h = 0.2
sigma = 0.5


denoised_img = non_local_means_denoise(img, h=h, sigma=sigma)

# denoised_img = denoise_nl_means(img, h=1.15 * sigma, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(92, 92, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Flatten())

model.add(Dense(67712))  # Reshape the tensor to (None, 67712)
model.add(Reshape((92, 92, 32)))  # Reshape the tensor to (None, 92, 92, 32)

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))


model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(img, img, batch_size=32, epochs=50, validation_split=0.2, callbacks=[checkpoint])

# Load the best model
model.load_weights('best_model.h5')

# Resize the image to the expected input shape of the model
img = resize(img, (92, 92, 3))

# Add an extra dimension to represent the batch size
img = np.expand_dims(img, axis=0)

# Predict the denoised image using the model
denoised_image = model.predict(img.squeeze())

# Save the denoised image
imsave('denoised_image.jpg', denoised_image[0])
print('Input data shape:', denoised_image.shape)
print('Input data type:', denoised_image.dtype)


# Denoise the image using the trained CNN model
# denoised_img = model.predict(np.expand_dims(denoised_img, axis=0))[0]

# Denoise the image using TV Chambolle algorithm
# denoised_img = denoise_tv_chambolle(denoised_img, weight=0.1, multichannel=True)

# Save the denoised image
# cv2.imwrite('denoised_image.jpg', denoised_img)






# Load the denoised and noisy images
# denoised_img = np.load('denoised_images.jpg')
# img = np.load('noisy_images.jpg')

# # Create a checkpoint callback to save the model weights after each epoch
# checkpoint = ModelCheckpoint('model_weights.h5', save_weights_only=True)

# # Print the shape of the input data
# print('Input data shape:', denoised_img.shape)

# # Print the data types of the input data and target data
# print('Input data type:', denoised_img.dtype)
# print('Target data type:', img.dtype)

# Train the model
# model.fit(denoised_img, img, batch_size=32, epochs=50, validation_split=0.2, callbacks=[checkpoint])




