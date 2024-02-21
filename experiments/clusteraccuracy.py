from sklearn.metrics.pairwise import euclidean_distances
import pickle
import math
import torch
from scipy.spatial.distance import pdist
import os
import tqdm
import numpy as np


def eval_initials(H_train, y_train, H_test, y_test, case = 'min'):
    
    y_pred = []
    y_true_ = []
    count = 0
    
    for xte, yte in zip(H_test, y_test):
        print(count)
        count += 1

        D_true = euclidean_distances(xte[np.newaxis,:], H_train[y_train==1]).squeeze()
        print(xte[np.newaxis,:].shape)
        print(H_train[y_train==1].shape)
        D_false = euclidean_distances(xte[np.newaxis,:], H_train[y_train==0]).squeeze()
            
        # Test if test points are correct
        if case=='min':
            # for test point to be correctly classified its distance to the nearest train sample must be smaller than to any other
            if np.min(D_true) <= np.min(D_false):
                y_pred.append(1)
            else:
                y_pred.append(0)
        elif case == 'mean':
            kn = 5
            if np.mean(sorted(D_true)[:kn]) <= np.mean(sorted(D_false)[:kn]): 
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_true_.append(yte)

    TP = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp==yt) and (yt==1.)])
    FN = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp!=yt) and (yt==1.)])

    TN = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp==yt) and (yt==0.)])
    FP = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp!=yt) and (yt==0.)])

    # Sensitivity, hit rate, recall, or true positive rate
    tpr = TP/(TP+FN) 
    
    # Fall out or false positive rate
    fpr = FP/(TN+FP) 
    
    # False negative rate
    fnr = FN/(TP+FN)
    
    # False discovery rate
    fdr = FP/(TP+FP)

    return (y_pred, y_true_, TP, FN, TN, FP, tpr, fpr, fnr, fdr)

SEEDS=[109706657, 253505912, 541217254, 506598379, 577336311]



defined_clusters = ['figureletters', 'mauresque', 'gothic']

data_reps = ["VGG16_Initials_0untrained_binary.pickle", "VGG16_Initials_-7untrained_binary.pickle", "VGG16_Initials_-14untrained_binary.pickle", "VGG16_Initials_Gram0default.pickle", "VGG16_Initials_Gram-7default.pickle", "VGG16_Initials_Gram-14default.pickle"]

data_reps_labels = ["0default", "-7default", "-14default", "0gram", "-7gram", "-14gram"]

out_dict = {c:{rep: [] for rep in data_reps_labels} for c in defined_clusters}

with open("/usr/users/anika.merklein/BA/"+ "images_list.pickle", 'rb') as f:
    i_list=pickle.load(f)

print(i_list)

for idx, data in enumerate(data_reps):
    with open("/usr/users/anika.merklein/BA/"+data, 'rb') as f:
        data=pickle.load(f)
    if type(data)==list:
            data = torch.stack([i.flatten() for i in data])
    for cluster in defined_clusters:
        with open("/usr/users/anika.merklein/BA/"+ cluster + ".pickle", 'rb') as f:
            cluster_data=pickle.load(f)
        print("clusterlen", len(cluster_data))
        bin_labels = []

        for i in i_list:
            if i in cluster_data:
                bin_labels.append(1)
            else:
                bin_labels.append(0)

        outp = []

        for seed in tqdm.tqdm(SEEDS):

            bin_labels = np.array(bin_labels)
            print(len(bin_labels), "bin_labels_len")

            # split set in pos and neg
            true_inds = np.where(bin_labels == 1)[0]
            false_inds = np.where(bin_labels == 0)[0]
            print(len(false_inds))
            np.random.seed(seed)
            np.random.shuffle(true_inds)
            np.random.shuffle(false_inds)
            
            split_ind_true = int(len(true_inds)/4.)*3
            split_ind_false = int(len(false_inds)/4.)*3

            inds_train_true, inds_train_false, inds_test_true, inds_test_false  = true_inds[:split_ind_true], false_inds[:split_ind_false], true_inds[split_ind_true:], false_inds[split_ind_false:]
            print(len(inds_train_true), len(inds_train_false), len(inds_test_true), len(inds_test_false))

            H_test, y_test = torch.cat((data[inds_test_true].squeeze(), data[inds_test_false].squeeze())), np.concatenate((bin_labels[inds_test_true].squeeze(), bin_labels[inds_test_false].squeeze()))
            H_train, y_train = torch.cat((data[inds_train_true].squeeze(), data[inds_train_false].squeeze())), np.concatenate((bin_labels[inds_train_true].squeeze(), bin_labels[inds_train_false].squeeze()))
            print(H_train.shape)
            outs = eval_initials(H_train, y_train, H_test, y_test, case='min')
            out_dict[cluster][data_reps_labels[idx]].append(outs)

with open("/usr/users/anika.merklein/BA/" +defined_clusters[0] + data_reps_labels[0] + ".pickle", 'wb') as f:
            pickle.dump(out_dict, f)


    
