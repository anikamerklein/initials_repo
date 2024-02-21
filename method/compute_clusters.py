from sklearn.cluster import KMeans
import numpy as np
import pickle
import torch


#load data

with open("Reps/"+"VGG16_Initials_Gram-14default_wlowtriag.pickle", 'rb') as f: 
    data=pickle.load(f)

if isinstance(data, list):
    transformed_list = [torch.flatten(i).reshape(1, 262144) for i in data]
    data = torch.cat(transformed_list, dim=0)

if isinstance(data, np.ndarray): 
    data = torch.from_numpy(data)


def compute_within_cluster_variance(X, c):
    var = torch.sum((X-c)**2)/X.shape[0]
    return var

def get_variances(X, C, y_pred):
    variances = []
    for i,c in enumerate(C):
        vari = compute_within_cluster_variance(X[y_pred==i], c)
        variances.append((i,vari))
    return variances


def get_single_distances(X, C, y_pred):
    distances = []
    for i,x in enumerate(X):
        c = C[y_pred[i]]
        dist = np.linalg.norm(x-c)
        distances.append(dist)
    return np.array(distances)


def cluster(X,K,seed, compute_vars_dsts = False):
    kmean = KMeans(n_clusters=K, random_state=seed)
    out = kmean.fit(X)
    y_pred = out.labels_
    intertia_cl  = out.inertia_
    if compute_vars_dsts:
        C = kmean.cluster_centers_
        variances = get_variances(X, C, y_pred)
        distances = get_single_distances(X,C,y_pred)
    
        return y_pred, intertia_cl, C, variances, distances, 
    else:
        return y_pred, intertia_cl




inertia = []

for k in np.arange(10, 110, 10):
    print(k)
    inertia.append(cluster(data, k, seed = 0, compute_vars_dsts=False))


with open("data/"+"VGG16_Initials_Gram_default_clusters_alldim_lowtriag_seed0"+".pickle", "wb") as f:
        pickle.dump(inertia, f)