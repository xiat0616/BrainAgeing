# -*- coding: utf-8 -*-
import datetime
t = str(datetime.datetime.now())

lr = 0.0001
decay=0.0001
age_dim = 100
age_ord = True
Norm = "BN"
latent_space = 130

EXPERIMENT_PARAMS = {
    "latent_space": latent_space,
    "self_rec_weight":0,
    "Norm":Norm,
    "expand_dim": 60,
    "age_ord": age_ord,
    "age_con": False,
    "age_dim": age_dim,
    "reg_subtract":False,
    "use_tanh":False,
    "l1_reg_weight":100,
    "data_dir":"Data/CamCAN", #The data folder, can change it to "Data/ADNI".
    "ncritic": [10,5], # Number of training critic
    "seed": 2019,  # Seed for the pseudorandom number generator.
    "folder": "experiments/method_age_AD_latent_space_%d_%s_age_ord_%s_%s_"%(latent_space,age_dim, age_ord, Norm)+t.split(' ')[0]+'-'+t.split(' ')[1].replace('.','-').replace(':','-'),  # Folder to store experiment results.
    "epochs": 600,
    "batch_size": 32,
    "split": 0,  # Default split of the dataset.
    "augment": True,  # Data augmentation
    "model": "synthesis_model_age_AD.synthesis_model_age_AD",  # Selected GAN architecture.
    "executor": "executor_age_AD.executor_age_AD",  # Selected experiment executor.
    "out_channels": 1,
    "gp_weight":10,
    "outputs": 1,
    "filters":32,
    # "num_masks": ACDCLoader().num_masks,  # Skip this, belongs to BaseModel.
    "lr": lr,  # Skip this, belongs to BaseModel.
    "beta_1":0,
    "beta_2":0,
    "decay": decay,  # Skip this, belongs to BaseModel.
    "input_shape":(160,208,1),
    "discr_params":{
        "latent_space": latent_space,
        "age_dim":age_dim,
        "depth":5,
        "input_shape":(160,208,1),
        "name":"discriminator",
        "filters":32,
        "lr": lr,  # Learning rate.
        "decay": decay,  # Decay rate.
    },
    "gen_params":{
        "latent_space": latent_space,
        "age_dim":age_dim,
        "depth":4,
        "input_shape": (160, 208, 1),
        "name": "generator",
        "filters": 32,
        "lr": lr,  # Learning rate.
        "decay": decay,  # Decay rate.
        "G_activation": 'linear'
    }
}


def get():
    return EXPERIMENT_PARAMS
