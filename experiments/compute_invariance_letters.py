import pickle
import numpy as np
import scipy.spatial
from sklearn.preprocessing import StandardScaler
import torch

ims = pickle.load(open('/usr/users/anika.merklein/BA/images_list.pickle', 'rb'))
labels = pickle.load(open('/usr/users/anika.merklein/BA/lettersdf.pickle', 'rb'))


layers = [-14, -7, 0]
all_data = {'gram' : {k:None for k in layers},
            'raw' : {k:None for k in layers}}


for case in ['gram', 'raw']:
    for l in layers:

        
        if case == 'gram':
            file = '/usr/users/anika.merklein/BA/VGG16_Initials_Gram{}untrained.pickle'.format(l)
        else:
            file = '/usr/users/anika.merklein/BA/VGG16_Initials_{}untrained_binary.pickle'.format(l)

        data = pickle.load(open(file, 'rb'))
        if type(data)==list:
            data = torch.stack([i.flatten() for i in data])
        print(case, l, data.shape)
        
        all_data[case][l] = data.numpy()
        
letters = [list(labels[labels.initial == im].letter)[0] for im in ims]
letters = np.array([l.replace('INIT_Letter_', '') for l in letters])

idxs = np.array(list(range(5862)))
np.random.shuffle(idxs)
rand_idxs = idxs[:2000]
standardize = ''

for metric in [ 'cosine', 'euclidean']:
    print(metric)
    dist_data = {}
    for case in ['gram', 'raw']:

        dist_data[case] = {l: {} for l in layers}


        for l in layers:
            data = all_data[case][l]
          
            if standardize == 'std':
                scaler = StandardScaler()
                scaler.fit(data)
                data = scaler.transform(data)
            
            dists_all = scipy.spatial.distance.pdist(data[rand_idxs,:], metric = metric) 
                
            for letter in list(set(letters)):

                print(letter, (letters==letter).sum())
                                
                data_letter = data[letters==letter]                        
                dists_letter = scipy.spatial.distance.pdist(data_letter, metric = metric)
                dist_data[case][l][letter] = (dists_letter.mean(), dists_all.mean(), (letters==letter).sum())

              #  if len(dist_data[case][l]) == 10:
              #      break     

    pickle.dump(dist_data, open('data_{}_{}.p'.format(metric, standardize), 'wb'))
    
    
    
    
