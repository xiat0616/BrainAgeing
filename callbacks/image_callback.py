# -*- coding: utf-8 -*-

import logging
import os
import matplotlib
matplotlib.use('agg')
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.age_ord_vector import get_age_ord_vector, calculate_age_diff

log = logging.getLogger("BaseSaveImage")


class ImageCallback_age(Callback):
    """Callback for saving training images."""
    def __init__(self, conf, model, comet_exp=None):
        super(ImageCallback_age, self).__init__()

        self.folder = os.path.join(conf.folder, "training_images")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.model = model
        self.comet_exp = comet_exp
        self.conf = conf

    def on_epoch_end(self, epoch=None, yng_img = None, yng_age=None, old_img = None, old_age=None):
        self.model.generator.save(self.folder + '/generator.h5df')
        self.model.discriminator.save(self.folder + '/critic.h5df')

        r, c = 4, 4
        diff_age = get_age_ord_vector(calculate_age_diff(yng_age, old_age), expand_dim=1, con=self.conf.age_con,
                                      ord=self.conf.age_ord, age_dim=self.conf.age_dim)
        gen_old_msk = self.model.generator.predict([yng_img, diff_age])
        gen_old_img = gen_old_msk+yng_img

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            axs[i, 0].set_title("Disc: %.1f" % self.model.discriminator.predict([yng_img[cnt:cnt + 1],old_age[cnt:cnt+1]]), size=6)
            axs[i, 0].imshow(np.concatenate([yng_img[cnt, :, :, j] for j in range(yng_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 0].axis("off")


            axs[i, 1].set_title("Disc: %.1f" % self.model.discriminator.predict([gen_old_msk[cnt:cnt+1],old_age[cnt:cnt+1]]), size=6)
            axs[i, 1].imshow(np.concatenate([gen_old_msk[cnt, :, :, j] for j in range(gen_old_msk.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 1].axis("off")

            axs[i, 2].set_title("Disc: %.1f" % self.model.discriminator.predict([gen_old_img[cnt:cnt+1],old_age[cnt:cnt+1]]), size=6)
            axs[i, 2].imshow(np.concatenate([gen_old_img[cnt, :, :, j] for j in range(gen_old_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 2].axis("off")


            axs[i, 3].set_title("Disc: %.1f" % self.model.discriminator.predict([old_img[cnt:cnt + 1], old_age[cnt:cnt+1]]), size=6)
            axs[i, 3].imshow(np.concatenate([old_img[cnt, :, :, j] for j in range(old_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 3].axis("off")
            cnt += 1

        fig.savefig(self.folder + "/gen_img_%d.png" % epoch)
        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure_name='gen_img_%d'%epoch, figure=fig )


class ImageCallback_age_AD(Callback):
    """Callback for saving training images."""
    def __init__(self, conf, model, comet_exp=None):
        super(ImageCallback_age_AD, self).__init__()

        self.folder = os.path.join(conf.folder, "training_images")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.model = model
        self.comet_exp = comet_exp
        self.conf = conf

    def on_epoch_end(self, epoch=None, yng_img = None, yng_age=None, yng_AD=None, old_img = None, old_age=None, old_AD=None):
        self.model.generator.save(self.folder + '/generator.h5df')
        self.model.discriminator.save(self.folder + '/critic.h5df')

        r, c = 4, 4
        diff_age = get_age_ord_vector(calculate_age_diff(yng_age, old_age), expand_dim=1, con=self.conf.age_con,
                                      ord=self.conf.age_ord, age_dim=self.conf.age_dim)
        gen_old_msk = self.model.generator.predict([yng_img, diff_age, old_AD])
        gen_old_img = gen_old_msk+yng_img

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            axs[i, 0].set_title("Disc: %.1f" % self.model.discriminator.predict([yng_img[cnt:cnt + 1], old_age[cnt:cnt+1], old_AD[cnt:cnt+1]]), size=6)
            axs[i, 0].imshow(np.concatenate([yng_img[cnt, :, :, j] for j in range(yng_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 0].axis("off")


            axs[i, 1].set_title("Disc: %.1f" % self.model.discriminator.predict([gen_old_msk[cnt:cnt+1],old_age[cnt:cnt+1], old_AD[cnt:cnt+1]]), size=6)
            axs[i, 1].imshow(np.concatenate([gen_old_msk[cnt, :, :, j] for j in range(gen_old_msk.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 1].axis("off")

            axs[i, 2].set_title("Disc: %.1f" % self.model.discriminator.predict([gen_old_img[cnt:cnt+1],old_age[cnt:cnt+1], old_AD[cnt:cnt+1]]), size=6)
            axs[i, 2].imshow(np.concatenate([gen_old_img[cnt, :, :, j] for j in range(gen_old_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 2].axis("off")


            axs[i, 3].set_title("Disc: %.1f" % self.model.discriminator.predict([old_img[cnt:cnt + 1], old_age[cnt:cnt+1], old_AD[cnt:cnt+1]]), size=6)
            axs[i, 3].imshow(np.concatenate([old_img[cnt, :, :, j] for j in range(old_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 3].axis("off")
            cnt += 1

        fig.savefig(self.folder + "/gen_img_%d.png" % epoch)
        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure_name='gen_img_%d'%epoch, figure=fig )


