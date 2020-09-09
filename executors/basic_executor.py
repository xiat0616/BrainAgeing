import logging
from abc import abstractmethod

import numpy as np


log = logging.getLogger("executor")


class Executor(object):
    def __init__(self, conf, model, comet_exp=None):
        self.conf = conf
        self.model = model
        self.epoch = 0
        self.models_folder = self.conf.folder + "/models"
        self.train_data = None
        self.valid_data = None
        self.train_folder = None
        self.experiment = None
        self.comet_exp = comet_exp
    @abstractmethod
    def init_train_data(self):
        pass

    @abstractmethod
    def get_loss_names(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        """
        Evaluate a model on the test data.
        """
        pass

    @abstractmethod
    def validate(self, epoch_loss):
        pass

    def stop_criterion(self, es, logs):
        pass

    def get_datagen_params(self):
        """
        Construct a dictionary of augmentations.
        :return: a dictionary of augmentation parameters to use with a keras image processor
        """
        if self.conf.augment:
            return dict(horizontal_flip=True, vertical_flip=False, rotation_range=0,
                        width_shift_range=0, height_shift_range=0) #fill_mode="constant", cval=cval)
        else:
            return dict()



