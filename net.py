# Usage : Directly use the model
# input : 512*512*3 images
# return : A list - [region score, afinity score]
# Dimensions : (h/2, w/2)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Input, Conv2D, Dense, MaxPool2D, Concatenate, BatchNormalization, UpSampling2D
from keras.applications.vgg16 import VGG16
from keras import Model


# Defines the upconv block
def upconv(n1, n2, f):
    x = Conv2D(n1, (1, 1), activation = "relu", padding = "same")(f)
    x = BatchNormalization()(x)
    x = Conv2D(n2, (3, 3), activation = "relu", padding = "same")(x)
    x = BatchNormalization()(x)
    return upsample(x)

# Upsamples the feature maps
def upsample(x):
    x = UpSampling2D((2, 2), interpolation = "bilinear")(x)
    return x;

# Loads pre-trained VGG16
vgg16 = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(512, 512, 3),
)

# Sets the weights as untrainable
vgg16.trainable = False


# Adds layers as per the requirement of the basenet
stage5 = vgg16.get_layer('block5_conv3').output

s61 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(stage5)
s62 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(s61)
stage6 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(s62)

u10 = Concatenate()([stage5, stage6])
up1 = upconv(512, 256, u10)

u20 = Concatenate()([up1, vgg16.get_layer('block4_conv3').output])
up2 = upconv(256, 128, u20)

u30 = Concatenate()([up2, vgg16.get_layer('block3_conv3').output])
up3 = upconv(128, 64, u30)

u40 = Concatenate()([up3, vgg16.get_layer('block2_conv2').output])
up4 = upconv(64, 32, u40)

ra1 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(up4)
ra2 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ra1)
ra3 = Conv2D(16, (3, 3), activation = "relu", padding = "same")(ra2)
ra4 = Conv2D(16, (3, 3), activation = "relu", padding = "same")(ra3)

region = Conv2D(1, (1, 1), activation = "sigmoid", padding = "same")(ra4)
affinity = Conv2D(1, (1, 1), activation = "sigmoid", padding = "same")(ra4)

model = Model(inputs = vgg16.input, outputs=[region, affinity])
model.summary()