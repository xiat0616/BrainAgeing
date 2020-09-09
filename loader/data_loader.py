import numpy as np
import os

class data_loader(object):
    """
    Data loader
    """
    def __init__(self, conf):
        self.data_folder = os.path.abspath(conf.data_dir)
        self.conf = conf

    def load_data(self, dataset="cam_can"):
        """
        return: all data and all labels
        """
        # Load data from saved numpy arrays
        imgs_yng_tr = np.load(self.data_folder+'/yng_imgs_tr.npy')
        age_yng_tr = np.load(self.data_folder+'/yng_ages_tr.npy')
        AD_yng_tr = np.load(self.data_folder+'/yng_AD_tr.npy') # For CamCAN AD vector is always healthy

        imgs_yng_te = np.load(self.data_folder+'/yng_imgs_te.npy')
        age_yng_te = np.load(self.data_folder+'/yng_ages_te.npy')
        AD_yng_te = np.load(self.data_folder+'/yng_AD_te.npy')

        imgs_old_tr = np.load(self.data_folder+'/old_imgs_tr.npy')
        age_old_tr = np.load(self.data_folder+'/old_ages_tr.npy')
        AD_old_tr = np.load(self.data_folder+'/old_AD_tr.npy') # For CamCAN AD vector is always healthy

        imgs_old_te = np.load(self.data_folder+'/old_imgs_te.npy')
        age_old_te = np.load(self.data_folder+'/old_ages_te.npy')
        AD_old_te = np.load(self.data_folder+'/old_AD_te.npy')

        return imgs_yng_tr, age_yng_tr, AD_yng_tr, imgs_yng_te, age_yng_te, AD_yng_te, \
               imgs_old_tr, age_old_tr, AD_old_tr, imgs_old_te, age_old_te, AD_old_te



