import cv2
import numpy as np
import logging
import os
import shutil

from matplotlib import cm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, Callback
from keras.utils import np_utils
import pandas as pd
#from ggplot import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class History(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(1)
        plt.subplot(211)
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='Training loss')
        # if loss_type == 'epoch':
        #     # val_acc
        #     plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        #     # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='Test loss')
        plt.xlabel(loss_type)
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.subplot(212)
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='Training: Acc')
        plt.plot(iters, self.val_acc[loss_type], 'b', label='Test acc')

        plt.xlabel(loss_type)
        plt.ylabel('Accuracy %')
        plt.grid(True)
        plt.legend(loc="upper right")

        plt.show()


def read_images_from_single_face_profile(face_profile, face_profile_name_index, dim = (48, 48)):
                     # face_profile: ../face_profiles/yaleBnn
    """
    Reads all the images from one specified face profile into ndarrays
    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile
    face_profile_name_index: int
        The name corresponding to the face profile is encoded in its index
    dim: tuple = (int, int)
        The new dimensions of the images to resize to
    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_one_face_profile, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all the images in the specified face profile
    Y_data : numpy array, shape = (number_of_images_in_face_profiles, 1)
        A face_profile_index data array contains the index of the face profile name of the specified face profile directory
    """
    X_data = np.array([])
    index = 0
    for the_file in os.listdir(face_profile):    #face_profile: ../face_profiles/yaleBnn
        file_path = os.path.join(face_profile, the_file)
        if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".pgm"):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img_data = img.ravel()
            X_data = img_data if not X_data.shape[0] else np.vstack((X_data, img_data))
            index += 1

    if index == 0 :
        shutil.rmtree(face_profile)
        logging.error("\nThere exists face profiles without images")

    Y_data = np.empty(index, dtype = int)                        #number of pictures in one yaleB file
    Y_data.fill(face_profile_name_index)    # Y_data: [face_profile_name_index,......,face_profile_name_index ]
                                                    # [i,i,i,i,..........................................i,i,i]
    return X_data, Y_data      # X_data: array shape=(number of pictures, pixels of picture)

def load_training_data(face_profile_directory):  #face_profile_directory   ../face_profiles/

    """
    Loads all the images from the face profile directory into ndarrays
    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory
    face_profile_names: list
        The index corresponding to the names corresponding to the face profile directory
    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_face_profiles, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all face_profiles
    Y_data : numpy array, shape = (number_of_face_profiles, 1)
        A face_profile_index data array contains the indexs of all the face profile names
    """

    # Get a the list of folder names in face_profile as the profile names
    face_profile_names = [d for d in os.listdir(face_profile_directory) if "." not in str(d)]
    # face_profile_names :yaleB01,yaleB02......
    if len(face_profile_names) < 2:
        logging.error("\nFace profile contains too little profiles (At least 2 profiles are needed)")
        exit()

    first_data = str(face_profile_names[0])
    first_data_path = os.path.join(face_profile_directory, first_data)   # first_data_path:../face_profiles/yaleB01
    X1, y1 = read_images_from_single_face_profile(first_data_path, 0)
    X_data = X1
    Y_data = y1
    print ("Loading Database: ")
    print (0, "    ",X1.shape[0]," images are loaded from:", first_data_path)
    for i in range(1, len(face_profile_names)):
        directory_name = str(face_profile_names[i])
        directory_path = os.path.join(face_profile_directory, directory_name)
        tempX, tempY = read_images_from_single_face_profile(directory_path, i)
        X_data = np.concatenate((X_data, tempX), axis=0)
        Y_data = np.append(Y_data, tempY)
        print (i, "    ",tempX.shape[0]," images are loaded from:", directory_path)

    return X_data, Y_data, face_profile_names       # X_data: (2452,2500), Y_data: (2452,)

# Load training data from face_profiles/
face_profile_data , face_profile_name_index , face_profile_names = load_training_data("./face_profiles/")
print("\n", face_profile_name_index.shape[0], " samples from ", len(face_profile_names), " people are loaded")

x = face_profile_data
y = face_profile_name_index
x_train,         x_test,    y_train,   y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)
#(1961, 2304) (491, 2304)   (1961,)    (491,)
# x_train=x_train.astype('float32')
# x_test=x_test.astype('float32')
# x_train //= 1.
# x_test //= 1.
# x_train= x_train.reshape(x_train.shape[0], 1, cmath.sqrt(x_train.shape[1]), cmath.sqrt(x_train.shape[1]))
# x_test=x_test.reshape(x_test.shape[0], 1, cmath.sqrt(x_test.shape[1], cmath.sqrt(x_test.shape[1])))
# x_train= x_train.reshape(x_train.shape[0], 1, 48, 48)/255
# x_test=x_test.reshape(x_test.shape[0], 1, 48, 48)/255

x_train= x_train.reshape(x_train.shape[0], 48, 48, 1)/255
x_test=x_test.reshape(x_test.shape[0], 48, 48, 1)/255
# y_train = np_utils.to_categorical(y_train,)
# y_test = np_utils.to_categorical(y_test,)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1961, 1, 48, 48) (491, 1, 48, 48)


#将标签转为one_hot类型
# def label_to_one_hot(labels_dense, num_classes=38):
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#     return labels_one_hot
#
# y_train=label_to_one_hot(y_train,38)
# y_test=label_to_one_hot(y_test,38)

batch_size = 128
epochs = 12

nb_classes = 38
input_shape=(48, 48, 1)
# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 36
# size of pooling area for max pooling
pool_size = (2, 2)
strides=(2, 2)
# convolution kernel size
kernel_size = (5, 5)
model=Sequential()
model.add(Convolution2D(nb_filters1, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=(48, 48, 1)))  # 卷积层1      output(16,48,48)
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling2D(pool_size=pool_size, strides=strides, padding='same'))  # pool 1  output(16,24,24)
model.add(Convolution2D(nb_filters2, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层2  output(36, 24, 24)
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling2D(pool_size=pool_size,strides=strides, padding='same'))  # pool 2   output(36,12,12)
model.add(Dropout(0.5))  # 神经元随机失活
model.add(Flatten())  # 拉成一维数据  36*12*12=5184
model.add(Dense(512))  # 全连接层1
model.add(Activation('relu'))  # 激活层
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('softmax'))  # Softmax评分
# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              #loss='categorical_crossentropy',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
model.summary()
# Another way to train the model
#model.fit(x_train, y_train, epochs=38, batch_size=400)
history = History()
#model.optimizer.lr.set_value(0.0005)
model.fit(x_train, y_train, batch_size=300, epochs=100, validation_data=(x_test, y_test), callbacks=[history])
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test,)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

#y_pred=model.predict(x_test)
y_pred=model.predict_classes(x_test)
confusion_mat = confusion_matrix(y_test, y_pred)
print("\nconfusion matrix=")
print(confusion_mat)
print("\n confusion matrix shape=")
print(confusion_mat.shape)
norm_conf = []
for i in confusion_mat:
    a=0
    tmp_arr= []
    a=sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)
plt.clf()
fig=plt.figure()
ax=fig.add_subplot(111)
res = ax.imshow(np.array(norm_conf), cmap=cm.jet, interpolation='nearest')
for i, cas in enumerate(confusion_mat):
    for j, c in enumerate(cas):
        if c > 0:
            plt.text(j - .5, i + .5, c, fontsize=6)

# df = pd.DataFrame(history.history)
# df.ix[:, 'Epoch'] = np.arange(1, len(history.history['acc'])+1, 1)
# p=pd.melt(df.ix[0:100, ['Iter','acc', 'val_acc']], id_vars='Iter')
#            # value_name=('Iter', 'value', color='variable')
# print(p)
history.loss_plot('epoch')
