from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from utils.age_ord_vector import get_age_ord_vector, calculate_age_diff
import numpy as np
import logging
# from configuration.exp_proposed_ADNI_long import EXPERIMENT_PARAMS
from keras.utils import Progbar
from executors.basic_executor import Executor
from loader.data_loader import data_loader
from easydict import EasyDict
from callbacks.loss_callback import SaveLoss
from callbacks.image_callback import ImageCallback_age as ImageCallback
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from callbacks.clr_callback import CyclicLR

log = logging.getLogger("proposed_executor")

class executor_age_AD(Executor):
    """
    This is the basic model where we put as input the old age to the generator in the middle.
    """
    def __init__(self, conf, model=None, comet_exp=None):
        super(executor_age_AD, self).__init__(conf, model, comet_exp)

    def init_train_data(self):
        loader = data_loader(self.conf)
        img_yng_tr, age_yng_tr , AD_yng_tr, _, _, _, img_old_tr, age_old_tr, AD_old_tr, _, _, _  = loader.load_data()

        self.train_img_yng = img_yng_tr[0:int((len(img_yng_tr) // self.conf.batch_size) * self.conf.batch_size)]
        self.train_age_yng = age_yng_tr[0:int((len(age_yng_tr) // self.conf.batch_size) * self.conf.batch_size)]
        self.train_AD_yng  = AD_yng_tr[0:int((len(AD_yng_tr) // self.conf.batch_size) * self.conf.batch_size)]

        self.train_img_old = img_old_tr[0:int((len(img_old_tr) // self.conf.batch_size) * self.conf.batch_size)]
        self.train_age_old = age_old_tr[0:int((len(age_old_tr) // self.conf.batch_size) * self.conf.batch_size)]
        self.train_AD_old  = AD_old_tr[0:int((len(AD_old_tr) // self.conf.batch_size) * self.conf.batch_size)]

        self.conf.data_len = len(self.train_img_yng)

    def get_loss_names(self):
        return [ "Discriminator_loss", "Discriminator_real_loss", "Discriminator_fake_loss",
                 "Generator_fake_loss", "Generator_l1_reg_loss", "Discriminator_gp_loss",
                 "Generator_zero_gre_loss"]

    def train(self):
        self.init_train_data()
        # make genetrated data
        gen_dict = self.get_datagen_params()

        # Here we need to concatenate age and AD labels, in order to use Function ImageDataGenerator
        yng_labels = np.concatenate([self.train_age_yng, self.train_AD_yng], axis=1)
        old_labels = np.concatenate([self.train_age_old, self.train_AD_old], axis=1)

        old_gen = ImageDataGenerator(**gen_dict).flow(x=self.train_img_old, y=old_labels, batch_size=self.conf.batch_size)
        yng_gen = ImageDataGenerator(**gen_dict).flow(x=self.train_img_yng, y=yng_labels, batch_size=self.conf.batch_size)

        # initialize training
        batches = int(np.ceil(self.conf.data_len/self.conf.batch_size))
        progress_bar = Progbar(target=batches * self.conf.batch_size)

        sl = SaveLoss(self.conf.folder)
        cl = CSVLogger(self.conf.folder+'/training.csv')
        cl.on_train_begin()
        img_clb = ImageCallback(self.conf, self.model, self.comet_exp)

        # clr = CyclicLR(base_lr=self.conf.lr/5, max_lr=self.conf.lr,
        #                step_size=batches*4, mode='triangular')

        loss_names = self.get_loss_names()
        total_loss = {n: [] for n in loss_names}

        # start training
        for epoch in range(self.conf.epochs):
            log.info("Train epoch %d/%d"%(epoch, self.conf.epochs))
            epoch_loss = {n: [] for n in loss_names}
            epoch_loss_list = []
            pool_to_print_old, pool_to_print_yng = [], []

            for batch in range(batches):
                old_img, old_labels = next(old_gen)
                yng_img, yng_labels = next(yng_gen)

                # Return labels to age and AD vectors
                old_age = old_labels[:,:self.conf.age_dim,:]
                old_AD  = old_labels[:,self.conf.age_dim:,:]

                yng_age = yng_labels[:,:self.conf.age_dim,:]
                yng_AD  = yng_labels[:,self.conf.age_dim:,:]

                if len(pool_to_print_old)<30:
                    pool_to_print_old.append(old_img)

                if len(pool_to_print_yng)<30:
                    pool_to_print_yng.append(yng_img)

                # Adversarial ground truths
                real_pred = -np.ones((old_img.shape[0],1))
                fake_pred = np.ones((old_img.shape[0],1))
                dummy = np.zeros((old_img.shape[0],1))
                dummy_Img = np.ones(old_img.shape)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                age_gap = calculate_age_diff(yng_age, old_age)
                diff_age = get_age_ord_vector(age_gap, expand_dim=1,con=self.conf.age_con,
                                              ord=self.conf.age_ord, age_dim=self.conf.age_dim)
                # Get a group of synthetic msks and imgs
                gen_masks = self.model.generator.predict([yng_img, diff_age, old_AD])
                gen_old_img = np.tanh(gen_masks+yng_img) if self.conf.use_tanh else gen_masks+yng_img
                # Need to train discriminators more iterations:
                if epoch<25:
                    for _ in range(self.conf.ncritic[0]):
                        epsilon = np.random.uniform(0, 1, size=(old_img.shape[0], 1, 1, 1))
                        interpolation = epsilon * old_img + (1 - epsilon) * gen_old_img
                        h_d = self.model.critic_model.fit([old_img, old_age, old_AD, gen_old_img, old_age, old_AD, interpolation, old_age, old_AD],
                                                          [real_pred, fake_pred, dummy], epochs=1, verbose=0)
                                                          # , callbacks=[clr])
                    # d_loss_bce = np.mean([h_real.history['binary_crossentropy'], h_fake.history['binary_crossentropy']])
                else:
                    for _ in range(self.conf.ncritic[1]):
                        epsilon = np.random.uniform(0, 1, size=(old_img.shape[0], 1, 1, 1))
                        interpolation = epsilon * old_img + (1 - epsilon) * gen_old_img
                        h_d = self.model.critic_model.fit([old_img, old_age, old_AD, gen_old_img, old_age, old_AD, interpolation, old_age, old_AD],
                                                          [real_pred, fake_pred, dummy], epochs=1, verbose=0)
                                                          # , callbacks=[clr])

                # d_loss_bce = np.mean(h_real.history['d_loss'])
                print('d_real_loss',np.mean(h_d.history['d_real_loss']),
                      'd_fake_loss', np.mean(h_d.history['d_fake_loss']))
                d_loss_bce = np.mean([h_d.history['d_real_loss'], h_d.history['d_fake_loss']])
                d_loss_real = np.mean(h_d.history['d_real_loss'])
                d_loss_fake = np.mean( h_d.history['d_fake_loss'])
                d_loss_gp = np.mean(h_d.history['gp_loss'])
                epoch_loss['Discriminator_loss'].append(d_loss_bce)
                epoch_loss['Discriminator_real_loss'].append(d_loss_real)
                epoch_loss['Discriminator_fake_loss'].append(d_loss_fake)
                epoch_loss['Discriminator_gp_loss'].append(d_loss_gp)
                # --------------------
                #  Train Generator
                # --------------------
                # Train the generator, want discriminator to mistake images as real
                h = self.model.gan.fit([yng_img, old_age, diff_age, age_gap, old_AD], [real_pred , dummy_Img], epochs=1, verbose=0)
                                       # , callbacks=[clr])
                # print(h.history)
                g_loss_bce = h.history['discriminator_loss']
                g_loss_l1 = h.history['map_l1_reg_loss']

                # Deal with epoch loss

                epoch_loss['Generator_fake_loss'].append(g_loss_bce)
                epoch_loss['Generator_l1_reg_loss'].append(g_loss_l1)

                #-----------------------------------------
                # Train Generator by self-regularization
                #-----------------------------------------
                diff_age_zero = yng_age-yng_age
                h = self.model.GAN_zero_reg([yng_img, diff_age_zero, yng_AD], yng_img , epochs=1, verbose=0)
                g_zero_reg = np.mean(h.history['self_reg'])
                epoch_loss['Generator_zero_gre_loss'].append(g_zero_reg)

                # Plot the progress
                progress_bar.update((batch+1)*self.conf.batch_size)

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))

            log.info(str('Epoch %d/%d: ' + ', '.join([l + ' Loss = %.3f' for l in loss_names])) %
                     ((epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            logs = {l: total_loss[l][-1] for l in loss_names}

            cl.model = self.model.discriminator
            cl.model.stop_training = False
            cl.on_epoch_end(epoch, logs)
            sl.on_epoch_end(epoch, logs)
            img_clb.on_epoch_end(epoch, yng_img, yng_age, old_img, old_age)

