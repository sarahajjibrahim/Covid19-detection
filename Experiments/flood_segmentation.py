# -*- coding: utf-8 -*-
"""
Created on Sat Oct  15 21:04:56 2022
@author: Sara
"""
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
import skimage
from glob import glob
import warnings
warnings.filterwarnings("ignore")

# %%
kernel_size = 5
iterations = 2

# %% Load images
def load_images(paths):
    images = []
    for path in paths:
        if "png" in path:
            image = imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            images.append(image)
    return images

# Plot images
def plot_images(arr, title=''):
    plt.figure(figsize=(15, 25))
    for i in range(len(arr)):
        plt.subplot(1, len(arr), i + 1)
        plt.title(title)
        plt.imshow(arr[i], cmap='gray')

# Merge image with binary mask
def mask_merge_segmented(uri_image, image_cluster):
    new_image = uri_image.copy()
    new_image[:, :] *= image_cluster
    return new_image

# Segment using flood technique
# https://stackoverflow.com/a/67896956/12899060
def flood(image):
    # Convert image to gray
    gray = rgb2gray(image)
    # Convert to float and divide by 255
    image_float = image.astype(float) / 255.
    # Calculate channel K and convert back to uint 8
    k_channel = 1 - np.max(image_float, axis=2)
    k_channel = (255*k_channel).astype(np.uint8)
    # Threshold via Otsu
    _, mask_binary = cv2.threshold(
        k_channel, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Get the structuring element
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Perform closing
    mask_binary = cv2.morphologyEx(
        mask_binary, cv2.MORPH_CLOSE, morph_kernel, None, None, iterations, cv2.BORDER_REFLECT101)
    # Merge binary mask with image
    image_result = mask_merge_segmented(gray, mask_binary)
    image_result = np.array(image_result)
    return gray, mask_binary, image_result

# %%
print(os.getcwd())
# %% Get images
# Download data https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database  
covid_images = glob('data/unsegmented/Covid/*')
normal_images = glob('data/unsegmented/Normal/*')
print("# covid images")
print(len(covid_images))
print("# normal images")
print(len(normal_images))
# %% Load some images
covid = load_images(covid_images[:5])
normal = load_images(normal_images[:5])
covid_1 = load_images(covid_images[:5])
normal_1 = load_images(normal_images[:5])
# %% Plot sample images
plot_images(covid_1)
plot_images(normal_1)
# %% Select any image and show plot its results, step by step segmentation
image_selected = covid[2]
gray, mask_binary, image_result = flood(image_selected)
# Find the contours on the binary image:
image_copy2 = image_selected.copy()
contours, hierarchy = cv2.findContours(
    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Look for the outer bounding boxes (no children):
for _, c in enumerate(contours):
    # Get the contours bounding rectangle:
    boundRect = cv2.boundingRect(c)
    # Get the dimensions of the bounding rectangle:
    rectX = boundRect[0]
    rectY = boundRect[1]
    rectWidth = boundRect[2]
    rectHeight = boundRect[3]
    # Set bounding rectangle:
    color = (255, 255, 0)
    cv2.rectangle(image_copy2, (int(rectX), int(rectY)),
                  (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
# %% Plot original image, binary mask, and resulting image after merge
plot_images([gray, image_copy2, mask_binary, image_result])
# %% Apply previous segmentation steps to all covid images
data_covid = list()
i = 0
for image_selected in covid:
    gray, mask_binary, image_result = flood(image_selected)
    data_covid.append(image_result)
# %% Apply previous segmentation steps to all normal images
data_normal = list()
for image_selected in normal:
    gray, mask_binary, image_result = flood(image_selected)
    data_normal.append(image_result)
# %% Length
len(data_normal)
len(data_covid)
# %% Create a directory to store segmented images
out_dir = "data/segmentation flood/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    os.makedirs(out_dir+"Normal/")
    os.makedirs(out_dir+"Covid/")
# %% Save covid images by index number
for k, image1 in enumerate(data_normal):
    save_path = os.getcwd()+"/data/segmentation flood/Normal/"
    skimage.io.imsave(os.path.join(save_path, "Normal-" +
                      str(k+1) + ".jpg"), img_as_ubyte(image1))
# %% Save normal images by index number
for c, image2 in enumerate(data_covid):
    save_path = os.getcwd()+"/data/segmentation flood/Covid/"
    skimage.io.imsave(os.path.join(save_path, "COVID-" +
                      str(c+1) + ".jpg"), img_as_ubyte(image2))
# %% Plot normal and Covid image in details
plt.imshow(data_normal[3], cmap='hot')
plt.imshow(data_covid[3], cmap='hot')
