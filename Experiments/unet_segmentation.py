# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:11:01 2022
# Same code imported from https://www.kaggle.com/code/limonhalder/lung-segmentation-using-u-net
@author: Sara
"""
import os
import PIL
import re
import cv2
import numpy as np
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import Model

# %% Prepare data (Train + Test)
def prepare_train_test(df=pd.DataFrame(), resize_shape=tuple(), color_mode="rgb"):
    img_array = list()
    mask_array = list()
    for image_path in tqdm(paths_df.image_path):
        resized_image = cv2.resize(cv2.imread(image_path), resize_shape)
        resized_image = resized_image/255.
        if color_mode == "gray":
            img_array.append(resized_image[:, :, 0])
        elif color_mode == "rgb":
            img_array.append(resized_image[:, :, :])
    for mask_path in tqdm(paths_df.mask_path):
        resized_mask = cv2.resize(cv2.imread(mask_path), resize_shape)
        resized_mask = resized_mask/255.
        mask_array.append(resized_mask[:, :, 0])
    return img_array, mask_array

# Test image for u-net segmentation
def test_on_image(model, img_array, img_num, img_side_size=256):
    pred = model.predict(img_array[img_num].reshape(
        1, img_side_size, img_side_size, 1))
    pred[pred > 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    plt.subplot(1, 2, 1)
    plt.imshow(pred.reshape(img_side_size, img_side_size), cmap="gray")
    plt.title("Prediction")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_array[img_num].reshape(
        img_side_size, img_side_size), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    return pred

# Get metrics of model
def get_metrics(history):
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Entropy")
    plt.subplot(2, 2, 2)
    plt.plot(history.history["coef"], label="training coefficient")
    plt.plot(history.history["val_coef"], label="validation coefficient")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Coef")

# Get coeff of model
def coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

# Get coeff loss
def coef_loss(y_true, y_pred):
    return -coef(y_true, y_pred)

# Construct unet model
def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
        2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

# %% Specify paths
# Download data for training unet: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
DIR = "data/Lung Segmentation/"
lung_image_paths = glob(os.path.join(DIR, "CXR_png/*.png"))
mask_image_paths = glob(os.path.join(DIR, "masks/*.png"))
# %% Get images
related_paths = defaultdict(list)
for img_path in lung_image_paths:
    img_path = img_path.replace("\\", "/")
    img_match = re.search(
        "CXR_png/([a-zA-Z0-9]*_[a-zA-Z0-9]*_[a-zA-Z0-9]*).png$", img_path)
    if img_match:
        img_name = img_match.group(1)
    for mask_path in mask_image_paths:
        mask_match = re.search(img_name, mask_path)
        if mask_match:
            related_paths["image_path"].append(img_path)
            related_paths["mask_path"].append(mask_path) 
            
# %% Get images
# Download data for testing: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database  
paths_df = pd.DataFrame.from_dict(related_paths)
DIR = "data/unsegmented/COVID/"
covid_image_paths = glob(os.path.join(DIR, "images/*.png"))
related_paths = defaultdict(list)
for covid_path in covid_image_paths:
    covid_match = re.search("COVID/(.*)\.png$", covid_path)
    if covid_match:
        covid_name = covid_match.group(1)
        related_paths["covid_path"].append(covid_path)
# %% Visual images
paths_dfc = pd.DataFrame.from_dict(related_paths)
xray_num = 9
img_path = paths_df["image_path"][xray_num]
mask_path = paths_df["mask_path"][xray_num]
img = PIL.Image.open(img_path)
mask = PIL.Image.open(mask_path)
# %% Show image samples
plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(img, cmap="gray")
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(mask, cmap="gray")
# %% Prepare data
img_array, mask_array = prepare_train_test(
    df=paths_df, resize_shape=(256, 256), color_mode="gray")
img_train, img_test, mask_train, mask_test = train_test_split(
    img_array, mask_array, test_size=0.2, random_state=42)
img_side_size = 256
img_train = np.array(img_train).reshape(
    len(img_train), img_side_size, img_side_size)
img_test = np.array(img_test).reshape(
    len(img_test), img_side_size, img_side_size)
mask_train = np.array(mask_train).reshape(
    len(mask_train), img_side_size, img_side_size)
mask_test = np.array(mask_test).reshape(
    len(mask_test), img_side_size, img_side_size)
# %% Construct U-net for segmentation 
model = unet(input_size=(256, 256, 1))
model.compile(optimizer=Adam(lr=5*1e-4), loss="binary_crossentropy",
              metrics=[coef, 'binary_accuracy'])
model.summary()
weight_path = "{}_weights.hdf5".format('cxr_reg')
checkpoint = ModelCheckpoint(weight_path, monitor='loss',
                             save_best_only=True,
                             save_weights_only=True)
early = EarlyStopping(monitor="loss",
                      patience=10)
callbacks_list = [checkpoint, early]
history = model.fit(x=img_train,
                    y=mask_train,
                    validation_data=(img_test, mask_test),
                    epochs=30,
                    batch_size=16,
                    callbacks=callbacks_list)
# %% Save model
model.save('/models/unet.h5')
np.save('/models/unet_history.npy', history.history)
# %% Test sample
IMG_NUM = 1
prediction = test_on_image(model, img_array=img_test,
                           img_num=IMG_NUM, img_side_size=256) 
get_metrics(history=history)
prediction = test_on_image(model, img_array=img_test,
                           img_num=IMG_NUM, img_side_size=256)
print(prediction)
