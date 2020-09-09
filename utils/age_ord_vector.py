import numpy as np

def synthesize_ordinal_age( age, con =True, age_dim=20, age_shape=None):
    stride = int(100//age_dim)
    age_vector = np.zeros((1,age_dim))
    age_l = int(np.floor(age/stride))
    age_res = age/stride-age_l

    if con:
        age_vector[:, 0:age_l]=1
        age_vector[:, age_l]=age_res
    else:
        age_vector[:, 0:age_l+1]=1

    # print(np.shape(age_vector))
    if not age_shape:
        return age_vector
    else:
        age_maps = np.zeros((1, age_shape[0], age_shape[1], age_dim))
        for i in range(age_dim):
            age_maps[:,:,:,i] = age_vector[:,i]
        return age_maps

def synthesize_one_hot_age(age, age_dim=20, age_shape=None):
    stride = int(100//age_dim)
    age_vector = np.zeros((1, age_dim))
    age_l = int(np.floor(age/stride))
    age_vector[:,age_l] = 1
    if not age_shape:
        return age_vector
    else:
        age_maps = np.zeros((1, age_shape[0], age_shape[1], age_dim))
        for i in range(age_dim):
            age_maps[:,:,:,i] = age_vector[:,i]
        return age_maps

def get_age_ord_vector(age, expand_dim=60, ord =True, con=False, age_dim=20, age_shape=None ):
    """
    Get age ord vector (optional)
    :param age: Age array, shape is (N,)
    :param expand_dim: Expand dims to (N * expand_dim, )
    :return: Age vector, (N * expand_dim, ) if not ord
                         (N * expand_dim, 20) if ord
    """
    age_temp = np.zeros((len(age), expand_dim))
    for l in range(len(age)):
        age_temp[l,:]=age[l]
    age_temp = np.reshape(age_temp, (-1,))

    if ord:
        age_vectors = np.zeros((len(age_temp),age_dim))
        for i in range(len(age_vectors)):
            age_vectors[i] = synthesize_ordinal_age(age_temp[i],con=con, age_dim=age_dim)
    else:
        age_vectors = np.zeros((len(age_temp),age_dim))
        for i in range(len(age_vectors)):
            age_vectors[i] = synthesize_one_hot_age(age_temp[i], age_dim=age_dim)
    return age_vectors

def get_age_continous_number(age, expand_dim=60, ord =True, con=False, age_dim=20, age_shape=None ):
    """
    Get age ord vector (optional)
    :param age: Age array, shape is (N,)
    :param expand_dim: Expand dims to (N * expand_dim, )
    :return: Age vector, (N * expand_dim, ) if not ord
                         (N * expand_dim, 20) if ord
    """
    age_temp = np.zeros((len(age), expand_dim))
    for l in range(len(age)):
        age_temp[l,:]=age[l]
    age_temp = np.reshape(age_temp, (-1,))

    age_continous_number = age_temp/100
    return age_continous_number


def calculate_age_diff(age_x, age_y, ord=True, con=False, age_dim=100):
    age_gap= 100//age_dim
    print(np.shape(age_x))
    age_diff = age_y-age_x
    age_diff = np.sum(age_diff, axis=1)
    return age_diff*age_gap

def get_age_from_age_vectors(age_v, age_dim=100):
    # age_dim = conf.dim
    stride = int(100//age_dim)
    age_values = np.sum(age_v, axis=-1)
    age_values = age_values*stride
    return  age_values-1

def get_age_as_continuous_values(age, expand_dim=60):
    """
    Get continuous age values
    """
    age_temp = np.zeros(shape=(len(age), expand_dim))
    for l in range(len(age)):
        age_temp[l,:]=age[l]
    age_temp = np.reshape(age_temp, (-1,))
    age_temp = age_temp/100
    return age_temp


def get_age_ord_maps(age_vector, expand_dim=60, ord =True, con=True, age_dim=20, age_shape=None ):
    """
    Get age ord maps
    :param age: Age array, shape is (N,)
    :param expand_dim: Expand dims to (N * expand_dim, )
    :return: Age maps, (N*expand_dim, ) if not ord
                         (N*expand_dim, 160, 208, 20) if ord
    """
    age_temp = np.zeros(shape=(len(age_vector), age_shape[0], age_shape[1], age_dim))
    # print(np.shape(age_temp))
    for l in range(len(age_vector)):
        for i in range(age_dim):
            # print(range(age_dim))
            age_temp[l,:,:,i]=age_vector[l,i]
    return age_temp

if __name__=="__main__":
    age = [12,20]
    age =get_age_as_continuous_values (age, expand_dim=10)
    print(age)
