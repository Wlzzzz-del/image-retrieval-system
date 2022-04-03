import scipy.io as io
from cv2 import imread
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import cv2
#实例化一个VGG16卷积基
#输入维度根据需要自行指定，这里仍然采用上一个例子的维度，卷积基的输出是(None,4,4,512)
###############单纯用VGG16卷积基直接提取特征，不使用图像增强####################
import os
from keras.preprocessing.image import ImageDataGenerator

# 图像迭代器
# 定义VGG16模型
def extract_feature(extractor, file):
    img = cv2.imread(file)
    img = cv2.resize(img, size,img)

    np_img_data = np.asarray(img)
    # 调整至适合模型的维度
    np_img_data = np.expand_dims(np_img_data,axis=0)

    # 提取特征并降维
    feature = extractor.predict(np_img_data)
    feature= np.reshape(feature,(4*4*512))
    return feature

conv_base = VGG16(weights="imagenet",include_top=False, input_shape=(150,150,3)) 
size = (150,150)
dir = "./ant"
feature_dic = dict()

for dirname,_,filesname in os.walk(dir):
    for i in filesname:
        path = dir+'/'+i
        feature = extract_feature(conv_base,path)
        feature_dic[path]=feature

io.savemat("a",feature_dic)
