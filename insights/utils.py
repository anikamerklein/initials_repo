import os
from skimage import io
from PIL import Image
import torchvision.transforms as transforms
import pickle
import numpy as np


def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_meta_initials(M_all):
    all_pages = [(os.path.split(d)[1].replace('.jpg',''), d) for d in M_all]

    all_pages = [p[0].replace('-1,', '') for p in all_pages]
    all_pages = [p.replace('-4,', '') for p in all_pages]
    all_pages = [p.replace('-', '_') for p in all_pages]
    
    years = [int(p.split("_")[-2]) for p in all_pages]
    #print(years)
    years = np.array(years)
    books = np.array(['_'.join((p.split('_')[:-2])) for p in all_pages])

    books_unique= sorted(list(set(books)))
    book_dict = {b:i for i,b in enumerate(books_unique)}

    return all_pages, years, books, books_unique,  book_dict



def get_hist(y, K):
    h = np.zeros(K)
    for y_ in y:
        h[y_]+=1
    return h
