# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 21:04:56 2022
@author: Sara
"""
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import os
import scipy
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.cluster import KMeans
import skimage
import cv2
from glob import glob
import warnings
warnings.filterwarnings("ignore")

# %%
clusters = 2

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

# Merge mask with image
def mask_merge_segmented(uri_image, image_cluster):
    new_image = uri_image.copy()
    new_image[:, :] *= image_cluster
    return new_image

# Mean filter
# https://stackoverflow.com/a/70541317/12899060
def mean_filter(arr, k):
    p = len(arr)
    diag_offset = np.linspace(-(k//2), k//2, k, dtype=int)
    eL = scipy.sparse.diags(np.ones((k, p)), offsets=diag_offset, shape=(p, p))
    nrmlize = eL @ np.ones_like(arr)
    return (eL @ arr) / nrmlize

# Select cluster index for image
# https://www.kaggle.com/code/naim99/image-classification-clustering-step-by-step
def select_cluster_index(clusters):
    minx = clusters[0].mean()
    index = 0
    for i in clusters:
        if i.mean() < minx:
            minx = i.mean()
            index += 1
    return index

# Thresholding
def binary(image):
    return image > threshold_otsu(image)

# Construct binary mask using k-means then segment
def k_means_cl(image):
    # Convert image to gray
    gray = rgb2gray(image)
    # Choose clusters = 2, as of colors, black and white, then construct binary mask via several steps
    k_means_image = KMeans(random_state=1,
                           n_clusters=clusters,
                           init='k-means++'
                           ).fit(gray.reshape((-1, 1))).labels_.reshape(gray.shape)
    clusters_colors_image = [k_means_image == i for i in range(clusters)]
    # Apply mean filter to the image
    image_mean_filter = mean_filter(
        clusters_colors_image[select_cluster_index(clusters_colors_image)], 20)
    mask_binary = binary(image_mean_filter)
    # Merge binary mask with image
    image_result = mask_merge_segmented(gray, mask_binary)
    return gray, mask_binary, image_result
 
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
image_selected = covid[0]
gray, test_binary, image_result = k_means_cl(image_selected)
# %% Plot original image, binary mask, and resulting image after merge
plot_images([image_selected, test_binary, image_result])
# %%
image_result.shape
# %% Apply previous segmentation steps to all covid images
data_covid = list()
i = 0
for image_selected in covid:
    gray, test_binary, image_result = k_means_cl(image_selected)
    data_covid.append(image_result)
# %% Apply previous segmentation steps to all normal images
data_normal = list()
for image_selected in normal:
    gray, test_binary, image_result = k_means_cl(image_selected)
    data_normal.append(image_result)
# %% Length
len(data_normal)
len(data_covid)
# %% Create a directory to store segmented images
out_dir = "data/segmentation kmeans/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    os.makedirs(out_dir+"Normal/")
    os.makedirs(out_dir+"Covid/")
# %% Save covid images by index number
for k, image1 in enumerate(data_normal):
    save_path = os.getcwd()+"/data/segmentation kmeans/Normal/"
    skimage.io.imsave(os.path.join(save_path, "Normal-" +
                      str(k+1) + ".jpg"), img_as_ubyte(image1))
# %% Save normal images by index number
for c, image2 in enumerate(data_covid):
    print(image2)
    save_path = os.getcwd()+"/data/segmentation kmeans/Covid/"
    skimage.io.imsave(os.path.join(save_path, "COVID-" +
                      str(c+1) + ".jpg"), img_as_ubyte(image2))
# %% Plot normal and Covid image in details
plt.imshow(data_normal[3], cmap='hot')
plt.imshow(data_covid[3], cmap='hot')
