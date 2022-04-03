from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Reshape,Dense,Flatten
from keras.applications.resnet50 import ResNet50
from keras.models import Model,Sequential
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

x_train_dir = './data/train/train'
noise_train_dir = './data/y_train'
test_dir = './data/test'
noise_test_dir = './data/noise_test'
train_datagen = ImageDataGenerator(rescale=1/255)
ytrain_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# 正常训练集迭代器
train_generator = train_datagen.flow_from_directory(
    x_train_dir,target_size=(300,200),batch_size=10
)

# 噪声训练集迭代器
"""
noise_train_generator = train_datagen.flow_from_directory(
    noise_train_dir,target_size=(300,200),batch_size=10
)
"""

# 正常测试集迭代器
test_generator = test_datagen.flow_from_directory(
    test_dir,target_size=(300,200),batch_size=10
)

# 噪声测试集迭代器
"""
noise_test_generator = test_datagen.flow_from_directory(
    noise_test_dir,target_size=(300,200),batch_size=10
)
"""

def resresnet():
    model = ResNet50()
    model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])
    model.fit_generator(
        train_generator, # 训练集生成器
        steps_per_epoch=10, # 每一轮训练生成两千个batch
        epochs=50, # 一共训练50轮
        validation_data=test_generator, # 验证集生成器
        validation_steps=10 # 验证集训练次数
        )
    pass

def my_model():
    input_img = Input(shape=(300,200,3))
    x = Conv2D(16,(3,3),activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2),padding = 'same')(x)
    x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2),padding = 'same')(x)
    x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2),padding = 'same', name = 'encoder')(x)

    """
    x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(8,(3,3),activation='relu', padding='same')(input_img)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(16,(3,3), activation='relu')(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

    """
    encoder = Model(input_img)
    encoder.compile(optimizer='adam', loss='binary_crossentropy')

    encoder.fit(train_generator,validation_data=test_generator)

    encoder.save("encoder_1.h5")

def mmodel():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=[5,5],padding='same',activation='relu',
    input_shape = (300,200,3)))
    model.add(MaxPooling2D(pool_size=[2,2],strides=2))
    model.add(Conv2D(filters=64,kernel_size=[5,5],padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=[2,2],strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))

    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss = "sparse_categorical_crossentropy",optimizer = "sgd",metrics = ["accuracy"])
    model.fit_generator(
        train_generator, # 训练集生成器
        steps_per_epoch=10, # 每一轮训练生成两千个batch
        epochs=50, # 一共训练50轮
        validation_data=test_generator, # 验证集生成器
        validation_steps=10 # 验证集训练次数
        )

my_model()
