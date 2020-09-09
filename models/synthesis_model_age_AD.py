from keras.optimizers import Adam
from keras.layers import Input, Add, Activation, Subtract
from keras.models import Model

from models.critics.critit_age_AD import critic_2D_with_AD
from models.generators.generator_age_AD import G_unet_16_2D_bn_with_AD
from models.basenet import BaseNet
import os
import logging
from utils.cost import wasserstein_loss, gradient_penalty_loss, l1_regularization, l1_regularization_loss
from functools import partial
# from configuration.exp_proposed_ADNI_involve_AD import EXPERIMENT_PARAMS
from easydict import EasyDict

log = logging.getLogger('ageing model')
# ************************************************************
# This is the extended model.
# ************************************************************

class proposed_method_involve_AD(BaseNet):
    def __init__(self, conf):
        super(proposed_method_involve_AD, self).__init__(conf)
        self.discriminator = None
        self.generator = None
        self.gan = None

    def build(self):
        # Build discriminator
        discr = critic_2D_with_AD(self.conf.discr_params)
        discr.build()

        log.info("Discriminator")
        self.discriminator = discr.model
        self.discriminator.summary(print_fn=log.info)

        # Build generator
        gen = G_unet_16_2D_bn_with_AD(self.conf.gen_params)

        gen.build()

        log.info("Generator")
        self.generator = gen.model
        self.generator.summary(print_fn=log.info)

        # In this combined model, we only train generator
        self.discriminator.trainable = False

        # The generator takes (young image and target age label) and outputs( a mask)
        X_yng= Input(shape=self.conf.input_shape)
        t_old = Input(shape=(self.conf.age_dim,))
        age_diff = Input(shape=(self.conf.age_dim,))
        age_gap = Input(shape=(1,))
        AD_vector = Input(shape=(self.conf.AD_dim,))

        M_old= self.generator([X_yng, age_diff, AD_vector ])

        # The synthetic old img = young img + addition mask
        gen_X_old = Add()([X_yng, M_old])
        # Use tanh or not?
        if self.conf.use_tanh:
            gen_X_old = Activation(activation='tanh')(gen_X_old)

        Map_reg = Activation(activation='linear', name="map_l1_reg")(M_old)
        # The discriminator takes (generated img, target age) and outputs prediction
        valid = self.discriminator([gen_X_old, t_old, AD_vector])

        partial_l1_regularization = partial(l1_regularization_loss,
                                            age_gap=age_gap)

        self.gan = Model(inputs=[X_yng, t_old, age_diff, age_gap, AD_vector], outputs=[valid ,Map_reg])
        self.gan.compile(loss={"discriminator":wasserstein_loss ,"map_l1_reg":partial_l1_regularization}, loss_weights=[1, self.conf.l1_reg_weight],
                         optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay))
        self.gan.summary(print_fn=log.info)

        # Make path to train discriminator
        self.discriminator.trainable = True
        real_old = Input(shape=self.conf.input_shape)
        real_age = Input(shape=(self.conf.age_dim,))
        real_AD  = Input(shape=(self.conf.AD_dim,))

        fake_old = Input(shape=self.conf.input_shape)
        fake_age = Input(shape=(self.conf.age_dim,))
        fake_AD  = Input(shape=(self.conf.AD_dim,))

        average_samples = Input(shape=self.conf.input_shape)
        average_age = Input(shape=(self.conf.age_dim,))
        average_AD  = Input(shape=(self.conf.AD_dim,))

        discriminator_real = Activation(activation='linear',name='d_real')(self.discriminator([real_old,real_age, real_AD]))
        discriminator_fake = Activation(activation='linear',name='d_fake')(self.discriminator([fake_old,fake_age, fake_AD]))
        discriminator_average = Activation(activation='linear',name='gp')(self.discriminator([average_samples,average_age, average_AD]))

        # The gradient penalty loss function requires the input averaged samples to get
        # gradients. However, Keras loss functions can only have two arguments, y_true and
        # y_pred. We get around this by making a partial() of the function with the averaged
        # samples here.
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=average_samples,
                                  gradient_penalty_weight=self.conf.gp_weight)
        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_old, real_age, real_AD, fake_old, fake_age, fake_AD, average_samples, average_age, average_AD],
                            outputs=[discriminator_real,
                                     discriminator_fake,
                                     discriminator_average])

        self.critic_model.compile(optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay),
                                loss={'d_real':wasserstein_loss,
                                      'd_fake':wasserstein_loss,
                                      'gp':partial_gp_loss})

        # ================================ Recreate image reg term=============================================
        X_yng = Input(shape=self.conf.input_shape)
        t_yng = Input(shape=(self.conf.age_dim,))
        t_AD  = Input(shape=(self.conf.AD_dim,))

        M_zero = self.generator([X_yng, t_yng, t_AD])
        gen_X_yng = Add(name="self_reg")([X_yng, M_zero])

        self.GAN_zero_reg = Model(inputs=[X_yng, t_yng, t_AD], outputs=gen_X_yng)
        self.GAN_zero_reg.compile(loss={"self_reg":"MAE"},
                                  loss_weights=[self.conf.self_rec_weight],
                                  optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay))
        self.GAN_zero_reg.summary(print_fn=log.info)

    def load_models(self):
        if os.path.exists(self.conf.folder + "/va_gan"):
            log.info("Loading trained model from file")
            self.gan.load_weights(self.conf.folder + "/va_gan")

    def save_models(self):
        log.debug("Saving trained model")
        self.gan.save_weights(self.conf.folder + "/va_gan")


if __name__=='__main__':

    pro_vaGAN = proposed_method_involve_AD(EasyDict(EXPERIMENT_PARAMS))
    pro_vaGAN.build()

