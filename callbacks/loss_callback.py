# -*- coding: utf-8 -*-

import os
import matplotlib
matplotlib.use('agg')
from keras.callbacks import Callback
import matplotlib.pyplot as plt


class SaveLoss(Callback):
    def __init__(self, folder, scale="linear"):
        super(SaveLoss, self).__init__()
        self.folder = folder
        self.values = dict()
        self.scale = scale

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return

        if len(self.values) == 0:
            for k in logs:
                self.values[k] = []

        for k in logs:
            self.values[k].append(logs[k])

        plt.figure()
        plt.suptitle("Training loss", fontsize=16)
        for k in self.values:
            epochs = range(len(self.values[k]))
            if self.scale == "linear":
                plt.plot(epochs, self.values[k], label=k)
            elif self.scale == "log":
                plt.semilogy(epochs, self.values[k], label=k)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.folder, "training_loss.png"))
        plt.close()