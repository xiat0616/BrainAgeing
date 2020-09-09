from keras.optimizers import Adam
from keras.layers import Input, Add, Activation
from keras.models import Model

from models.critics.critic_age import critic_2D
from models.generators.generator_age import G_unet_16_2D_bn
from models.basenet import BaseNet
import os
import logging
from utils.cost import wasserstein_loss, gradient_penalty_loss, l1_regularization_loss
from functools import partial

log = logging.getLogger('brain_ageing')
# ************************************************************
# This is the model for MICCAI paper
# ************************************************************

class synthesis_model_age(BaseNet):
    def __init__(self, conf):
        super(synthesis_model_age, self).__init__(conf)
        self.discriminator = None
        self.generator = None
        self.gan = None

    def build(self):
        # Build discriminator
        discr = critic_2D(self.conf.discr_params)
        discr.build()

        log.info("Discriminator")
        self.discriminator = discr.model
        self.discriminator.summary(print_fn=log.info)

        # Build generator
        gen = G_unet_16_2D_bn(self.conf.gen_params)
        gen.build()

        log.info("Generator")
        self.generator = gen.model
        self.generator.summary(print_fn=log.info)

        # In this combined model, we only train generator
        self.discriminator.trainable = False

        # The generator takes (young image and target age label) and outputs( a mask)
        X_yng= Input(shape=self.conf.input_shape)
        age_diff = Input(shape=(self.conf.age_dim,))
        t_old = Input(shape=(self.conf.age_dim,))
        age_gap = Input(shape=(1,))

        M_old= self.generator([X_yng, age_diff])

        # The synthetic old img = young img + addition mask
        gen_X_old = Add()([X_yng, M_old])

        # Use tanh or not?
        if self.conf.use_tanh:
            gen_X_old = Activation(activation='tanh')(gen_X_old)

        Map_reg = Activation(activation='linear', name="map_l1_reg")(M_old)
        # The discriminator takes (generated img, target age) and outputs prediction
        valid = self.discriminator([gen_X_old, t_old])

        partial_l1_regularization = partial(l1_regularization_loss,
                                            age_gap=age_gap)

        self.gan = Model(inputs=[X_yng, t_old, age_diff, age_gap], outputs=[valid ,Map_reg])
        self.gan.compile(loss={"discriminator":wasserstein_loss ,"map_l1_reg":partial_l1_regularization}, loss_weights=[1, self.conf.l1_reg_weight],
                         optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay))
        self.gan.summary(print_fn=log.info)

        # Make path to train discriminator
        self.discriminator.trainable = True
        real_old = Input(shape=self.conf.input_shape)
        real_age = Input(shape=(self.conf.age_dim,))
        fake_old = Input(shape=self.conf.input_shape)
        fake_age = Input(shape=(self.conf.age_dim,))
        average_samples = Input(shape=self.conf.input_shape)
        average_age = Input(shape=(self.conf.age_dim,))

        discriminator_real = Activation(activation='linear',name='d_real')(self.discriminator([real_old,real_age]))
        discriminator_fake = Activation(activation='linear',name='d_fake')(self.discriminator([fake_old,fake_age]))
        discriminator_average = Activation(activation='linear',name='gp')(self.discriminator([average_samples,average_age]))

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=average_samples,
                                  gradient_penalty_weight=self.conf.gp_weight)
        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_old, real_age, fake_old, fake_age, average_samples, average_age],
                            outputs=[discriminator_real,
                                     discriminator_fake,
                                     discriminator_average])

        self.critic_model.compile(optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay),
                                loss={'d_real':wasserstein_loss,
                                      'd_fake':wasserstein_loss,
                                      'gp':partial_gp_loss}
                                         )

    def load_models(self):
        if os.path.exists(self.conf.folder + "/ageing_model"):
            log.info("Loading trained model from file")
            self.gan.load_weights(self.conf.folder + "/ageing_model")

    def save_models(self):
        log.debug("Saving trained model")
        self.gan.save_weights(self.conf.folder + "/ageing_model")

