import pickle
from torch.nn.functional import normalize
import pandas as pd
import torch
import numpy as np

import sys

print(pd. __version__)


layer = int(sys.argv[1]) # layer from input

data = ["/usr/users/anika.merklein/BA/"+"VGG16_Initials_"+ str(layer)+"trained_binary.pickle", "/usr/users/anika.merklein/BA/"+"VGG16_Initials_"+ str(layer)+ "untrained_binary.pickle"]

labels = ['trained', 'default']


# import letter data

with open("/usr/users/anika.merklein/BA/"+"lettersdf.pickle", 'rb') as f: 
    lettersdf=pickle.load(f)


letters = lettersdf['letter'].unique() 

# import images list

with open("/usr/users/anika.merklein/BA/"+"images_list.pickle", 'rb') as f: 
    images=pickle.load(f)


def compute_gram_matrix(input): 

    a, b = input.size()  

    
    features = input.reshape((a, 512, int(b/512)))


    gram_mtcs = []
    
    for i in features:
        print(i.shape) 
        G = torch.mm(i, i.t())  # compute the gram product
        G = G.div(a * b) # normalization
        gram_mtcs.append(G)

    return gram_mtcs

for idx, d in enumerate(data): 
    with open(d, 'rb') as f: 
        outcat=pickle.load(f)
    
    outcat = normalize(outcat, dim = 0)

    # calculate gram
    gram = compute_gram_matrix(outcat)

    # save gram rep
    with open("/usr/users/anika.merklein/BA/"+"VGG16_Initials_Gram"+str(layer)+str(labels[idx])+ ".pickle", "wb") as f:
        pickle.dump(gram, f)

