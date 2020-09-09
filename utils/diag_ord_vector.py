import numpy as np

def synthesize_ordinal_diag()


def get_diag_ord_vector(diag, ord =True, diag_dim=2, expand_dim=60):
    diag_number = np.zeros(shape=np.shape(diag))
    for l in range(len(diag_number)):
        if diag[l]=="CN": diag_number[l]=0
        if diag[l]=="MCI": diag_number[l]=1
        if diag[l]=="AD": diag_number[l]=2

    diag_temp = np.zeros(shape=(len(diag), expand_dim))
    for l in range(len(diag)):
        diag_temp[l,:]=diag_number[l]
    diag_temp = np.reshape(diag_temp,newshape=(-1,))

    # Generate ord vecors
    age_vectors = np.zeros((len(diag_temp), diag_dim))

    for l in len(diag_temp):

    return diag_temp


if __name__=="__main__":
    diag = ["AD", "MCI", "CN", "CN"]
    print(get_diag_ord_vector(diag, expand_dim=5))

