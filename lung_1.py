import os
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
#肺部数据链接
path_lung = 'G:/阿里天池/肺和结肠癌的组织病理学影像/lung_image_sets/lung_image_sets'
#结肠癌链接
path_colon = 'G:/阿里天池/肺和结肠癌的组织病理学影像/colon_image_sets/colon_image_sets'

# path_lung_aca_img = []
# path_lung_aca_label = []  #标签1
#
# path_lung_n_img = []
# path_lung_n_label = []#标签2
#
# path_lung_scc_img = []
# path_lung_scc_label = []#标签3
# #
# path_colon_aca_img = []
# path_colon_aca_label = []#标签4
#
# path_colon_n_img = []
# path_colon_n_label = []#标签5

img_list = []
img_label = []

def get_file_1(file_dir):
    for file in os.listdir(file_dir+'/'+'lung_aca'):
        img_list.append(file_dir+'/'+'lung_aca'+'/'+file)
        img_label.append(1)
    # for file in os.listdir(file_dir+'/'+'lung_n'):
    #     img_list.append(file_dir+'/'+'lung_n'+'/'+file)
    #     img_label.append(2)
    #
    # for file in os.listdir(file_dir + '/' + 'lung_scc'):
    #     img_list.append(file_dir + '/' + 'lung_scc' + '/'+file)
    #     img_label.append(3)

get_file_1(path_lung)
print(len(img_list))
# def get_file_2(file_dir):
#     for file in os.listdir(file_dir+'/'+'colon_aca'):
#         img_list.append(file_dir+'/'+'colon_aca'+'/'+file)
#         img_label.append(4)
#     for file in os.listdir(file_dir+'/'+'colon_n'):
#         img_list.append(file_dir+'/'+'colon_n'+'/'+file)
#         img_label.append(5)
# get_file_2(path_colon)
# print(img_list)
# image_list = np.hstack((path_lung_scc_img, path_lung_aca_img, path_lung_n_img, path_colon_aca_img,path_colon_n_img))
# label_list = np.hstack((path_lung_scc_label, path_lung_aca_label, path_lung_n_label, path_colon_aca_label,path_colon_n_label))
# temp = np.array([image_list, label_list])
# temp = temp.transpose()
# np.random.shuffle(temp)

# all_image_list_1 = list(temp[:, 0])
# all_label_list_1 = list(temp[:, 1])
def img_deal(img_list):
    total_image_img_list = []
    for i in img_list:
        img = io.imread(i)
        img = transform.resize(img, (64, 64))
        img = img/255.0
        img = img.astype('float16')
        total_image_img_list.append(img)
    return total_image_img_list
x = img_deal(img_list[:100])
print(x)
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(img_label)
y = to_categorical(y,5)
x = np.array(x)
#
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# print(x_train.shape)
# print(x_test.shape)
#
# from keras.preprocessing.image import ImageDataGenerator
# augs_gen = ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     zoom_range=0.1,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=False
# )
# augs_gen.fit(x_train)
