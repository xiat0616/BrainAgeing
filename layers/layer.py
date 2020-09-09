#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tian
"""
from keras.layers import BatchNormalization
from keras.layers import Activation, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers.merge import Add
from layers.InstanceNormalization import InstanceNormalization


def residual_block(l0, filters):
    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',  kernel_initializer="he_normal")(l0)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(l0)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    return Add()([l0, l])

def residual_block_in(l0, filters):
    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',  kernel_initializer="he_normal")(l0)
    l = InstanceNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(l0)
    l = InstanceNormalization()(l)
    l = Activation('relu')(l)
    return Add()([l0, l])

def g_upsampling(l0, filters):
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(filters=filters, kernel_size=7,  strides=1, padding='same',kernel_initializer="he_normal")(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    return l

def g_upsampling_in(l0, filters):
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(filters=filters, kernel_size=7,  strides=1, padding='same',kernel_initializer="he_normal")(l)
    l = InstanceNormalization()(l)
    l = Activation('relu')(l)
    return l

def conv2D_layer_bn(l0, name=None, filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                    kernel_initializer="he_normal"):
    l = Conv2D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                # activation="linear",
                activation=activation,
                kernel_initializer=kernel_initializer)(l0)
    l = BatchNormalization()(l)
    # l = Activation(activation=activation)(l)
    return l

def conv2D_layer_in(l0, name=None, filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                    kernel_initializer="he_normal"):
    l = Conv2D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                # activation="linear",
                activation=activation,
                kernel_initializer=kernel_initializer)(l0)
    l = InstanceNormalization()(l)
    # l = Activation(activation=activation)(l)
    return l

def conv2D_layer(l0, name=None, filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                    kernel_initializer="he_normal"):
    l = Conv2D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    # l = BatchNormalization()(l)
    return l

def deconv2D_layer_bn(l0, name=None, filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_initializer="he_normal"):
    l = Conv2DTranspose( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                         activation=activation,
                # activation="linear",
                kernel_initializer=kernel_initializer)(l0)
    l = BatchNormalization()(l)
    # l = Activation(activation=activation)(l)
    return l

def deconv2D_layer_in(l0, name=None, filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_initializer="he_normal"):
    l = Conv2DTranspose( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                        # activation="linear",
                         activation=activation,
                         kernel_initializer=kernel_initializer)(l0)
    l = InstanceNormalization()(l)
    l = Activation(activation=activation)(l)
    return l

def deconv2D_layer(l0, name=None, filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_initializer="he_normal"):
    l = Conv2DTranspose( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    return l

def conv3D_layer_bn(l0, name=None, filters=32, kernel_size=(3,3,3), strides=(1,1,1), padding=3, activation='relu',
                    kernel_initializer="he_nomral"):
    l = Conv3D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    return l

def deconv3D_layer_bn(l0, name=None, filters=32, kernel_size=(2,2,2), strides=(2,2,2), padding='same', activation='relu',
                      kernel_initializer="he_normal"):
    l = Conv3DTranspose( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    l = BatchNormalization()(l)
    return l

# """
# ============================================================================================
# Copy from link: https://blog.csdn.net/googler_offer/article/details/79521453
# ============================================================================================
# """
# def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
#     if name is not None:
#         bn_name = name + '_bn'
#         conv_name = name + '_conv'
#     else:
#         bn_name = None
#         conv_name = None
#
#     x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
#     x = BatchNormalization(axis=3, name=bn_name)(x)
#     return x
#
#
# def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
#     x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
#     x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
#     if with_conv_shortcut:
#         shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
#         x = Add()([x, shortcut])
#         return x
#     else:
#         x = Add()([x, inpt])
#         return x
#
