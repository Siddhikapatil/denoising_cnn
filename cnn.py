import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.callbacks import ModelCheckpoint
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle

# Load the noisy image
img = cv2.imread('noisy_imgg.jpg')
img = img_as_float(img)

# Denoise the image using Non-Local Means algorithm
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoised_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img.shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(denoised_img, img, batch_size=32, epochs=50, validation_split=0.2, callbacks=[checkpoint])

# Load the best model
model.load_weights('best_model.h5')

# Denoise the image using the trained CNN model
denoised_img = model.predict(np.expand_dims(denoised_img, axis=0))[0]

# Denoise the image using TV Chambolle algorithm
denoised_img = denoise_tv_chambolle(denoised_img, weight=0.1, multichannel=True)

# Save the denoised image
cv2.imwrite('denoised_image.jpg', denoised_img)