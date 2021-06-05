import os
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#肺部数据链接
path_lung = 'G:/阿里天池/肺和结肠癌的组织病理学影像/lung_image_sets/lung_image_sets'
#结肠癌链接
path_colon = 'G:/阿里天池/肺和结肠癌的组织病理学影像/colon_image_sets/colon_image_sets'

path_lung_aca_img = []
path_lung_aca_label = []  #标签1

path_lung_n_img = []
path_lung_n_label = []#标签2

path_lung_scc_img = []
path_lung_scc_label = []#标签3
#
path_colon_aca_img = []
path_colon_aca_label = []#标签4

path_colon_n_img = []
path_colon_n_label = []#标签5


def get_file_1(file_dir):
    for file in os.listdir(file_dir+'/'+'lung_aca'):
        path_lung_aca_img.append(file_dir+'/'+'lung_aca'+'/'+file)
        path_lung_aca_label.append(1)
    for file in os.listdir(file_dir+'/'+'lung_n'):
        path_lung_n_img.append(file_dir+'/'+'lung_n'+'/'+file)
        path_lung_n_label.append(2)
    for file in os.listdir(file_dir + '/' + 'lung_scc'):
        path_lung_scc_img.append(file_dir + '/' + 'lung_scc' + '/'+file)
        path_lung_scc_label.append(3)
get_file_1(path_lung)

def get_file_2(file_dir):
    for file in os.listdir(file_dir+'/'+'colon_aca'):
        path_colon_aca_img.append(file_dir+'/'+'colon_aca'+'/'+file)
        path_colon_aca_label.append(4)
    for file in os.listdir(file_dir+'/'+'colon_n'):
        path_colon_n_img.append(file_dir+'/'+'colon_n'+'/'+file)
        path_colon_n_label.append(5)
get_file_2(path_colon)

all_image_list_1 = []
all_image_list_1.extend(path_lung_aca_img[:200])
all_image_list_1.extend(path_lung_n_img[:200])
all_image_list_1.extend(path_lung_scc_img[:200])
all_image_list_1.extend(path_colon_aca_img[:200])
all_image_list_1.extend(path_colon_n_img[:200])

all_label_list_1 = []
all_label_list_1.extend(path_lung_aca_label[:200])
all_label_list_1.extend(path_lung_n_label[:200])
all_label_list_1.extend(path_lung_scc_label[:200])
all_label_list_1.extend(path_colon_aca_label[:200])
all_label_list_1.extend(path_colon_n_label[:200])

image_list = np.hstack(all_image_list_1)
label_list = np.hstack(all_label_list_1)
temp = np.array([image_list, label_list])
temp = temp.transpose()
np.random.shuffle(temp)
all_image_list_1 = list(temp[:, 0])
all_label_list_1 = list(temp[:, 1])
print(all_label_list_1)


total_image_img_list = []
for i in all_image_list_1:
    img = io.imread(i)
    img = transform.resize(img, (100, 100))
    img = img/255.0
    img = img.astype('float16')
    total_image_img_list.append(img)
print(total_image_img_list)
print(len(all_label_list_1))
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(all_label_list_1)
y = to_categorical(y,5)
x = np.array(total_image_img_list)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
print(x_train.shape)
print(x_test.shape)

from keras import layers
from keras import models
from keras import optimizers
cnn = models.Sequential()
cnn.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.5))
cnn.add(layers.MaxPooling2D((2,2)))
cnn.add(layers.Conv2D(32,(3,3),activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.5))
cnn.add(layers.MaxPooling2D((2,2)))
cnn.add(layers.Flatten())
cnn.add(layers.Dropout(0.5))
cnn.add(layers.Dense(16,activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dense(5,activation='softmax'))
cnn.compile(loss='categorical_crossentropy',
 optimizer=optimizers.Adam(lr=0.0003),
 metrics=['accuracy'])

history = cnn.fit(x_train,y_train,batch_size=10,validation_data=(x_test,y_test),epochs=30,verbose=1)

def show_history(history): #显示训练过程学习曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) +1)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='训练损失')
    plt.plot(epochs, val_loss, 'b', label='验证损失')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='训练正确精度')
    plt.plot(epochs, val_acc, 'b', label='验证正确精度')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
show_history(history)
history.save('./model_save/lung_colon.h5')


