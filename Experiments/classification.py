# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 13:41:49 2022
@author: Sara
"""
import os
import skimage
import joblib
import timeit
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from skimage.transform import resize
from sklearn.metrics import f1_score
from sklearn.utils import Bunch
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPooling2D

# %% Load files
def load_image_files(container_path, dimension=(28, 28)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir()
               if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "X-ray image dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        # read images in order
        d = sorted(os.listdir(direc), key=getint)
        print(direc)
        for file in d:
            file = ""+str(direc)+"/"+file+"" 
            img = skimage.io.imread(file)
            img_resized = resize(
                img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    images.reshape((-1, 1, 28, 28))

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
# Get number within image name to read images in order
def getint(name):
    _, num = name.split('-')
    if 'png' in num:
        num, _ = num.split('.png')
    if 'jpg' in num:
        num, _ = num.split('.jpg')
    return int(num)  

# Tune images for prediction 
def tune_images(images):
    images_transformed = []
    for i in images:
        i = i.reshape(28, 28)
        img_float32 = np.float32(i)
        imgRGB = cv2.cvtColor(img_float32, cv2.COLOR_BGRA2RGB)
        images_transformed.append(imgRGB)
    return np.asarray(images_transformed)

# %% Load dataset
image_dataset = load_image_files(os.getcwd()+"/data/unsegmented/")
#image_dataset = load_image_files(os.getcwd()+"/data/segmented_unet/")
#image_dataset = load_image_files(os.getcwd()+"/data/segmented_kmeans/")
#image_dataset = load_image_files(os.getcwd()+"/data/segmented_flood/")
# %%
print(image_dataset)
# %% Split dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.2, random_state=109)
# %% Get shapes
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# %% Predict images using RandomForestClassifier
model = RandomForestClassifier()
start = timeit.default_timer()
model.fit(x_train, y_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
joblib.dump(model, "models/RandomForestClassifier.joblib")
#model = load('models/RandomForestClassifier.joblib')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Predict images using KNeighborsClassifier
model = KNeighborsClassifier()
start = timeit.default_timer()
model.fit(x_train, y_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
joblib.dump(model, "models/KNeighbors.joblib")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Predict images using DecisionTreeClassifier
model = DecisionTreeClassifier()
start = timeit.default_timer()
model.fit(x_train, y_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
joblib.dump(model, "models/DecisionTreeClassifier.joblib")
#model = load('models/RandomForestClassifier.joblib')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Predict images using lgb
d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['boosting_type'] = 'gbdt'
start = timeit.default_timer()
model = lgb.train(params, d_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
y_pred = y_pred.round(0)
y_pred = y_pred.astype(int)
joblib.dump(model, "models/LGBClassifier.joblib")
#model = load('models/LGBClassifier.joblib')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Predict images using GradientBoostingClassifier
model = GradientBoostingClassifier()
start = timeit.default_timer()
model.fit(x_train, y_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
joblib.dump(model, "models/GradientBoostingClassifier.joblib")
#model = load('models/GradientBoostingClassifier.joblib')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Predict images using XGBClassifier
model = XGBClassifier()
start = timeit.default_timer()
model.fit(x_train, y_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
joblib.dump(model, "models/XGBClassifier.joblib")
#model = load('models/XGBClassifier.joblib')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Predict images using SVC
model = SVC(kernel='linear')
start = timeit.default_timer()
model.fit(x_train, y_train)
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
y_pred = model.predict(x_test)
joblib.dump(model, "models/SVC.joblib")
#model = load('models/SVC.joblib')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average=None))
print("AUC = ", roc_auc_score(y_test, y_pred))
# %% Construct CNN
model_chkpt = ModelCheckpoint(
    'best_mod.h5', save_best_only=True, monitor='accuracy')
# early stopping for preventing overfitting
early_stopping = EarlyStopping(
    monitor='loss', restore_best_weights=False, patience=10)
# Define a Sequential() model.
model = Sequential()
# Add the first layer: 32 is the number of filters; kernel_size specifies the size of our filters;
# activation specifies the activation function;input_shape specifies what type of input we are going to pass to the network
model.add(Conv2D(32, kernel_size=(3, 3),
          activation="relu", input_shape=(28, 28, 3)))
# Second layer: specified 64 filters(must be a power of 2)
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
# Deine a Max pooling: kernel_size, which specified the size of the pooling window.
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout. This means that the model will not overfit, as some neurons randomly will not be selected for activation.
# This prevents the model from overfitting.
model.add(Dropout(0.25))
# Repeate the above steps to make a deeper network.
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Flatten layer
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
# Create an output sigmoid function
model.add(Dense(1, activation="sigmoid"))
# Compile the model: binary_crossentropy because this is a binary classification problem; adam as the optimizer; the metric that we want to monitor is accuracy.
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
# Printe the model architecture to take a look at the number of parameters that the model will learn.
model.summary()
# %% Reshape data to fit model
x_test_cnn = tune_images(x_test)
x_train_cnn = tune_images(x_train)
# %% Predict images using CNN
start = timeit.default_timer()
history = model.fit(x_train_cnn, y_train,
                    validation_split=0.10,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    callbacks=[model_chkpt, early_stopping]
                    )
stop = timeit.default_timer()
print('Training time: ', stop - start, ' secs')
model.save('models/cnn.h5')
np.save('models/cnn_history.npy', history.history)
# %% Plot loss
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
# %%
y_pred = model.predict(x_test_cnn, batch_size=32)
label = [int(p >= 0.5) for p in y_pred]
# %%
print("Confusion Matrix:")
print(confusion_matrix(y_test, label))
print("Classification Report")
print(classification_report(y_test, label))
print("Accuracy = ", accuracy_score(y_test, label))
print("F1 score = ", f1_score(y_test, label, average=None))
print("AUC = ", roc_auc_score(y_test, label))
