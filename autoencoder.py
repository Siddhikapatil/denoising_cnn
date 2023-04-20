import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from denoise_img import non_local_means_denoise
# import main

# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from skimage.restoration import denoise_nl_means, estimate_sigma

# Load the noisy and clean images
noisy_img = cv2.imread('girl.jpg', 0)
# input_image = cv2.imread('girl.jpg')
clean_image = cv2.imread('output_image.jpg', 0)

# Normalize the pixel values to [0, 1]
# noisy_img = noisy_img / 255.0
# clean_image = clean_image / 255.0

# Add Gaussian noise to the noisy image
# mean = 0
# var = 0.01
# sigma_g = var * 0.5
# h=0.2
# noisy_img = noisy_img + np.random.normal(mean, sigma_g, noisy_img.shape)

# Apply Non-Local Means denoising to the noisy image
# sigma_est= denoise_tv_chambolle(noisy_img, multichannel=False)

# sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))
# patch_kw = dict(patch_size=5,      # 5x5 patches
#                 patch_distance=6,  # 13x13 search area
#                 multichannel=False)
# denoised_img = denoise_nl_means(noisy_img, h=1.15 * sigma_est, **patch_kw)

# Prepare the data for training the CNN-based autoencoder
output_image = non_local_means_denoise(noisy_img, h=0.1, sigma=1.0)
x_train = np.expand_dims(output_image, axis=-1)
y_train = np.expand_dims(clean_image, axis=-1)

# Define the architecture of the CNN-based autoencoder
input_img = Input(shape=(None, None, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
history = autoencoder.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Apply the trained autoencoder to the denoised image
denoised_img = np.expand_dims(output_image, axis=-1)
clean_pred = autoencoder.predict(denoised_img)
clean_pred = np.squeeze(clean_pred)

# Display the results
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Clean Image', clean_image)
cv2.imshow('Denoised Image (NLM)', output_image)
cv2.imshow('Denoised Image (CNN)', clean_pred)
cv2.waitKey(0)