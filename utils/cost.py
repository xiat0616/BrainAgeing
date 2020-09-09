import keras.backend as K
import numpy as np

def l1_regularization(y_true, y_pred):
    return K.mean(K.abs(y_pred))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def l1_regularization_loss(y_true, y_pred,  age_gap, age_range=60):

    epsilon= K.exp(-age_gap/age_range)
    # epsilon =1
    l1_loss = epsilon * K.mean(K.abs(y_pred-y_true), axis=(1,2,3))

    return K.mean(l1_loss)

def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=1)+0.1
    union = K.sum(y_true, axis=1)+K.sum(y_pred, axis=1)+0.1
    dice_coef = K.mean(2*intersection/union, axis=0)
    return dice_coef

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true,y_pred)

if __name__=="__main__":
    x = np.ones(shape=(3,3,3,3))
    y = np.ones(shape=(3,3,3,3))*1.1
    print(l1_regularization_loss(x, y, age_gap=30))