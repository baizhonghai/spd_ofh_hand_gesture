import random
import numpy as np


def read_random_list():
    random_seed = 36
    random.seed(random_seed)
    random_integers = np.array([random.randint(1, 100) for _ in range(20)])
    return random_integers
    #return read_random_list_1()

def read_random_list_1():
    random_integers = np.array([i for i in range(100)])
    return random_integers


'''if __name__=='__main__':
    read_random_list_1()'''