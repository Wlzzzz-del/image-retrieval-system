from keras.applications.vgg19 import VGG19
import ft
conv_base = VGG19(weights="imagenet",include_top=False, input_shape=(150,150,3),pooling='avg') 
t = ft.extract_feature(conv_base,"data\chair (1).jpg")
print(t.shape)

