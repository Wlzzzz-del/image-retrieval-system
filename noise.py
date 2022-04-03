import os
import cv2
import numpy as np
import random

# 源图像存储文件
x_train_dir = './data/test/test'
# 训练图像存储文件
y_train_dir = './data/noise_test/noise_test'

# 添加噪声的脚本
def add_noise(x_train_dir, y_train_dir, noise=0.5):
    img = cv2.imread(x_train_dir)
    output = np.zeros(img.shape, np.uint8)
    thre = 1-noise
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if noise <thre:
                output[i][j]=0
            elif noise > thre:
                output[i][j]=255
            else:
                output[i][j]=img[i][j]
    cv2.imwrite(y_train_dir,output)

for dir,_,files in os.walk(x_train_dir):
    for i in files:
        x = str(x_train_dir)+'/'+str(i)
        y = str(y_train_dir)+'/'+str(i)
        add_noise(x,y)
