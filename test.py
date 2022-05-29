from multiprocessing import Pool
import tqdm
import time
import numpy as np


# def _foo(my_number):
#    square = my_number * my_number
#    time.sleep(1)
#    return square
#
# if __name__ == '__main__':
#    with Pool(2) as p:
#       r = list(tqdm.tqdm(p.imap(_foo, range(10)), total=10))
#    print(r)