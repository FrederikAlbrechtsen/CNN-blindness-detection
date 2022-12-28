from keras import backend as K
from keras import optimizers
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import img_to_array
from tqdm import tqdm
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

base_dir = "blindness-detection/data/"

dataset = []
labels = []

def preprocess_img(label,path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img_array = img_to_array(img)
    img_array = img_array/255
    dataset.append(img_array)
    labels.append(str(label))

train_Data = pd.read_csv(base_dir +"/train.csv")
train_Data.head()

id_code_Data = train_Data['id_code']
diagnosis_Data = train_Data['diagnosis']

for id_code,diagnosis in tqdm(zip(id_code_Data,diagnosis_Data)):
    path = os.path.join(base_dir+'train','{}.png'.format(id_code))
    preprocess_img(diagnosis,path)

images = np.array(dataset)
label_arr = np.array(labels)

x_train,x_test,y_train,y_test = train_test_split(images,label_arr,test_size=0.10,random_state=42)

y_train = np_utils.to_categorical(y_train, num_classes=5)
y_test = np_utils.to_categorical(y_test, num_classes=5)

model = Sequential()
model.add(Conv2D(16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5,activation="softmax"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint("cnn.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
hist = model.fit(x_train,y_train,batch_size=64,epochs=30,validation_data=(x_test, y_test), callbacks=[checkpoint, early])

pred = model.predict(x_test)
score = round(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)),2)
print("Accuracy: ", score*100,"%")

report = classification_report(y_test.argmax(axis=1), pred.argmax(axis=1))
print(report)