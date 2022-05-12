import scipy.io as io
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import ft
#实例化一个VGG16卷积基
#输入维度根据需要自行指定，这里仍然采用上一个例子的维度，卷积基的输出是(None,4,4,512)
###############单纯用VGG16卷积基直接提取特征，不使用图像增强####################
import os

conv_base = VGG16(weights="imagenet",include_top=False, input_shape=(150,150,3)) 

mat = io.loadmat("a.mat")
feature1 = ft.extract_feature(conv_base,"./data/train/train/ant/image_0001.jpg")
#feature2 = ft.extract_feature(conv_base,"ant/barrel_2.jpg")
dis_dict = dict()

for i in mat:
    try:
        dis = mat[i]-feature1
        dis_dict[i] = np.linalg.norm(dis)
    except:pass
    # 待检测图像
# 排序
dis_dict = sorted(dis_dict.items(),key=lambda x:x[1],reverse=False)
k=0
for i in dis_dict:
    if(k<10):
        print(i)
        k +=1