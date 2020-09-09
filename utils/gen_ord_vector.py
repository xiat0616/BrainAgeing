import numpy as np
def get_gen_ord_vector(gen, gen_dim = 40, expand_dim=60):
    """
    Get gender vector

    input: (N, 1) gender
    return: (N* expand_dim. gen_dim) gender vector
    """
    gen_v = gen
    gen_v = np.repeat(gen_v, expand_dim)
    gen_v = np.expand_dims(gen_v, axis=-1)
    gen_v = np.repeat(gen_v, gen_dim, axis=-1)
    return gen_v

if __name__=="__main__":
    gen = [0,1]
    gen_vector = get_gen_ord_vector(gen, expand_dim=2, gen_dim=5)
    print(np.shape(gen_vector))
    print(gen_vector)


