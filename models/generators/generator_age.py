from keras.layers import  Dense, Input, MaxPool2D
from keras.layers import Reshape, BatchNormalization, Flatten
from models.basenet import BaseNet
from keras.models import Model

from layers.layer import  conv2D_layer_bn
from layers.layer import deconv2D_layer_bn
from keras.layers.merge import Concatenate
from easydict import EasyDict
# from configuration.exp_proposed import EXPERIMENT_PARAMS


class G_unet_16_2D_bn(BaseNet):
    def __init__(self, conf):
        super(G_unet_16_2D_bn, self).__init__(conf)

    def build(self, _subname=None):
        f       = self.conf.filters
        name    = self.conf.name + _subname if _subname else self.conf.name
        g_input = Input(shape=self.conf.input_shape)

        # (batch size, 160, 208, 1)
        conv1_1 = conv2D_layer_bn(g_input, name="conv1_1", filters=f, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv1_2 = conv2D_layer_bn(conv1_1, name="conv1_2", filters=f, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool1 = MaxPool2D()(conv1_2) # (batch size, 80, 104, filters)

        # (batch size, 80, 104, filters)
        conv2_1 = conv2D_layer_bn(pool1, name="conv2_1", filters=f*2, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv2_2 = conv2D_layer_bn(conv2_1, name="conv2_2", filters=f*2, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool2 = MaxPool2D()(conv2_2) # (batch size, 40, 52, filters*2)

        #  (batch size, 40, 52, filters*2)
        conv3_1 = conv2D_layer_bn(pool2, name="conv3_1", filters=f*4, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv3_2 = conv2D_layer_bn(conv3_1, name="conv3_2", filters=f*4, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool3 = MaxPool2D()(conv3_2) # (batch size, 20, 26, filters*4)

        # (batch size, 20, 26, filters*4)
        conv4_1 = conv2D_layer_bn(pool3, name="conv4_1", filters=f*8, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv4_2 = conv2D_layer_bn(conv4_1, name="conv4_2", filters=f*8, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool4 = MaxPool2D()(conv4_2)
        # (batch size, 10, 13, filters*8)
# ======================================================================================================================
        # In middle layer, we concatenate Age vector
        # (batch size, 10, 13, filters*8)
        mid1_1 = conv2D_layer_bn(pool4, name="mid1_1", filters=f, kernel_size=3, strides=1, padding='same',
                                 activation='relu', kernel_initializer="he_normal")
        # (batch size, 10, 13, filter)
        flat1_1 = Flatten()(mid1_1)
        # (batch size, 4160)
        dens1_1 = Dense(units=self.conf.latent_space, name="dens1_1",activation='sigmoid')(flat1_1)
        dens1_1 = BatchNormalization()(dens1_1)
        age_vector = Input(shape=(self.conf.age_dim,)) # Age vector shape: (20,)
        # (batch size, 130)
        mid_concat1_1 = Concatenate()([dens1_1, age_vector]) # Encode age vector into middle layer
        # (batch size, 130+100)
        dens2_1 = Dense(units=4160, name="dens2_1", activation='relu')(mid_concat1_1)
        # (batch size, 4160)
        rshape1_1 = Reshape(target_shape=(10,13,32))(dens2_1)
        # (batch size, 10, 13, 32)
        mid_concat2_1 = Concatenate()([pool4, rshape1_1])
        # (batch size, 10, 13, 32+f*8)
# ======================================================================================================================
        upconv4 = deconv2D_layer_bn(mid_concat2_1, name="upconv4", filters=f*8, kernel_size=3, strides=2, padding='same',
                           activation='relu', kernel_initializer="he_normal")
        concat4 = Concatenate()([upconv4, conv4_2])

        conv_m_1_1 = conv2D_layer_bn(concat4, name='conv_m_1_1', filters=f*8, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv_m_1_2 = conv2D_layer_bn(conv_m_1_1, name='conv_m_1_2', filters=f*8, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")

        # (batch size, 20, 26, filters*8)
        upconv3 = deconv2D_layer_bn(conv_m_1_2, name="upconv3", filters=f*4, kernel_size=3, strides=2, padding='same',
                                    activation='relu', kernel_initializer="he_normal")

        # (batch size, 40, 52, filters*4)
        concat3 = Concatenate()([upconv3, conv3_2])
        # (batch size, 40, 52, filters*8)

        conv5_1 = conv2D_layer_bn(concat3, name='conv5_1', filters=f*4, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv5_2 = conv2D_layer_bn(conv5_1, name='conv5_2', filters=f*4, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        # (batch size, 40, 52, filters*4)

        upconv2 = deconv2D_layer_bn(conv5_2, name="upconv2", filters=f*2, kernel_size=3, strides=2, padding='same',
                           activation='relu', kernel_initializer="he_normal" )
        # (batch size, 80, 104, filters*2)
        concat2 = Concatenate()([upconv2, conv2_2])
        # (batch size, 80, 104, filters*4)

        conv6_1 = conv2D_layer_bn(concat2, name='conv6_1',filters=f*2, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv6_2 = conv2D_layer_bn(conv6_1, name='conv6_2', filters=f*2, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        # (batch size, 80, 104, filters*2)
        upconv1 = deconv2D_layer_bn(conv6_2, name="upconv1", filters=f, kernel_size=3, strides=2, padding='same',
                           activation='relu', kernel_initializer="he_normal")
        # (batch size, 160, 208, filters)
        concat1 = Concatenate()([upconv1, conv1_2])
        # (batch size, 160, 208, filters*2)

        conv8_1 = conv2D_layer_bn(concat1, name="conv8_1", filters=f, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv8_2 = conv2D_layer_bn(conv8_1, name="conv8_2", filters=1, kernel_size=3, strides=1, padding='same',
                           activation=self.conf.G_activation, kernel_initializer="he_normal")
        # (batch size, 160, 208, 1)
        self.model = Model(inputs=[g_input, age_vector], outputs=conv8_2, name = name)
        self.model.summary()