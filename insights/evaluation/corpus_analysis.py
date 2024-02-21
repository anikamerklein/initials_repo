import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn.metrics
import glob
import tqdm as tqdm
import collections
import os
import itertools
import scipy
import scipy.stats as stats
from sklearn.cluster import KMeans

def compute_within_cluster_variance(X, c):
    var = np.sum((X-c)**2)/X.shape[0]
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
    y_pred = kmean.fit_predict(X)
    
    if compute_vars_dsts:
        C = kmean.cluster_centers_
        variances = get_variances(X, C, y_pred)
        distances = get_single_distances(X,C,y_pred)
    
        return y_pred, C, variances, distances, 
    else:
        return y_pred, C

def normal(x, t_center, t_sigma):
    return stats.norm.pdf(x, t_center, t_sigma)



def get_data_gausswindow(years, H_all, M_all, orig_inds, books, plot=False, t_sigma=5, n_samples =100, tstart = 1478, tend=1650):
    
    bin_centers = np.arange(tstart, tend+1)
    
    if plot:
        f,ax = plt.subplots(1,1)
    data = {}

    scats = []
    sampled = []
    for t_center in bin_centers: #[:-3]:

        t_idx = int(np.where(bin_centers==t_center)[0])        
        p_=normal(bin_centers, t_center, t_sigma)

        p_all = {k:v for k,v in  zip(bin_centers,p_)}
        p_all = np.array([p_all[t_] for t_ in years])
        p_all[np.logical_and(years>=t_center+t_sigma, years<t_center-t_sigma)] = 0.
        p_all = p_all/np.sum(p_all)
        

        out = np.random.choice(list(range(len(years))), size=n_samples, replace=False, p=p_all)
            
        t_max = collections.Counter(years[out]).most_common()[0][0]
        t_min_sample = np.min(years[out])
        t_max_sample = np.max(years[out])

        if plot:
            ax.scatter(t_center, t_max, c='grey', alpha=0.5)
        
        scats.append([t_center, t_max])
        sampled.append(orig_inds[out])
        
        mask = out
        Ht = H_all[mask]
        Bt = books[mask]
        Mt = np.array(M_all)[mask]
        data[(t_center,t_center)] = (Ht, Bt, Mt)

    if plot:
        ax.set_xlim([1494, 1647+1])
        ax.set_ylim([1494, 1647+1])
        ax.set_xlabel('center t')
        ax.set_ylabel('max sampled t')
        ax.plot(bin_centers, bin_centers, c='r')

    return data, scats, sampled




# do everythin only on the filtered data....

def get_evolve_historic_data_threshed(M, data, years, y_pred, thresh, K, plot=False ):
    cl_pages = {k:v for k,v in zip(M, y_pred)}
    times = sorted(data.keys(), key= lambda x: x[0])

    for nearest in [None]:
        evolve_data = {}
        for metric in ['euclidean']:

            
            if plot:
                f,axs = plt.subplots(2,1, figsize=(4,3), gridspec_kw={'height_ratios':[0.7,0.3]})

            sims = []
            counts = []

            entropy = []
            entropy2 = []
            entropy_cl = []

            e_vec = np.zeros(K)
            CLS_t = []
            P_t = []
            for t in times[:-1]:

                e_vec2 = np.zeros(K)

                Ht_ = data[t][0] # data points of timeslot

                Pt =  data[t][2] # names of images

                cls_t =  [cl_pages[x] for x in Pt]
                p_t = [p_ for p_ in Pt]
                cls_years = [years[y_pred==c] for c in cls_t] 

                #print(cls_t)


                for c_  in cls_t:
                    #print(c_)
                    e_vec[c_]+=1
                    e_vec2[c_]+=1

                #print(len(cls_years))
                entropy.append(scipy.stats.entropy(e_vec)) 
                entropy2.append(scipy.stats.entropy(e_vec2)) 
                #difference between e_vec and e_vec2: e_vec2 per timeslot. e_vec in total

                CLS_t.append(cls_t)
                P_t.append(p_t)

                counts.append(len(Ht_))


            evolve_data[(thresh, metric)] = (times, entropy, entropy2, CLS_t, P_t, counts) 

            if plot:

                axs[0].plot([t[0] for t in times[:-1]], entropy)
                axs[0].plot([t[0] for t in times[:-1]], entropy2, color='r')

                axs[1].plot([t[0] for t in times[:-1]], counts)
                axs[0].set_title(thresh)
                axs[0].set_title(thresh)
                axs[0].set_xticklabels(['']*len(list(axs[0].get_xticklabels())))

            if plot:
                plt.show()
        
        
    return evolve_data


def entropy_exp(H_all,M_all, books, years_input, n_sims = 1, n_samples = 70, cl_dict = None,
                Ks  = [500, 1000, 1500, 2000], 
                thresholds = [0,100,200,250,300], sample_sizes = [20, 40, 80, 150], H_cluster = None, compute_clusters = False, 
                t_sigma = 5):

        
    seed=1

    

    if isinstance(cl_dict, str):
        print('Computing from given clusters')
        cl_dict = pickle.load(open(cl_dict, 'rb'))
        cl_dict = {Ks[0]: cl_dict[4]} ## quickfix for clusters computed via compute for different cluster numbers

    
    elif compute_clusters ==True:
        print('Computing clusters')
        cl_dict = {}
        for K in Ks:
            y_pred2, centers,  variances, distances = cluster(H_all, K=K, seed=seed, compute_vars_dsts=True)
            cl_dict[K] = (y_pred2, centers,  variances, distances )
    
    elif H_cluster is not None:
        print('Computing random clusters')
        assert H_cluster is not None
        cl_dict = {}
        for K in Ks:
            y_pred2, centers,  variances, distances = cluster(H_cluster, K=K, seed=seed, compute_vars_dsts=True)
            cl_dict[K] = (y_pred2, centers,  variances, distances )


    else:
        raise

    colors = ['red', 'blue', 'black', 'cyan']
    all_inds = np.array(range(len(H_all)))
    handles = []
    all_sampled = {}


    all_data_cl = {}

    for l,K in enumerate(Ks):
        print(K)
        y_pred2, centers,  variances, distances  = cl_dict[K]

        cl_centers = {k:v for k,v in enumerate(centers)}
        all_samples = {sample:[] for sample in sample_sizes}
        for j in range(n_sims): # nsims
            data_hist_evolution = {}
            visited = []
            for ii, sample_size in enumerate(sample_sizes):
                
                H_filt = H_all
                M_filt = M_all

                years_filt, books_filt = years_input, books
                y_pred_filt = y_pred2
                orig_inds = all_inds

                    

                data, _, sampled_ = get_data_gausswindow(years_filt, H_filt, M_filt, orig_inds, books_filt, 
                                                         t_sigma= t_sigma, n_samples=sample_size, plot=True) 

                #get all the data that is new in each timestep
                entries_new_in_timestep = [0]*len(sampled_)
                for cnt,i in enumerate(sampled_):
                        if cnt == 0:
                            entries_new_in_timestep[0] = sampled_[cnt]
                            visited = sampled_[cnt]
                        else:
                            toadd = np.setdiff1d(sampled_[cnt], sampled_[cnt-1])
                            toadd = np.setdiff1d(toadd, visited)
                            entries_new_in_timestep[cnt] = sampled_[cnt]

                # save lists to be able to evaluate:

                with open('new_entries_per_timestemp_'+str(sample_size)+'.pickle', 'wb') as f:
                    pickle.dump(entries_new_in_timestep, f)
                    
                
                hilf =  get_evolve_historic_data_threshed(M_filt, data, years_filt, y_pred_filt, sample_size, K, plot=False)

   
                data_hist_evolution.update(hilf)

                all_samples[sample_size].append(sampled_)
        

            # collecting data        
            if j ==0:
                all_data = {k:[None,[], [], [], []] for k in data_hist_evolution.keys()}

            for k in all_data.keys():
                hilf = data_hist_evolution[k]
                times2 = [t[0] for t in hilf[0][:-1]]
                entropy2 = hilf[2]
                counts = hilf[5]
                clusters_t = hilf[3]
                pages_t = hilf[4]

                all_data[k][0]=times2
                all_data[k][1].append(entropy2)
                all_data[k][2].append(counts)

                if len(all_data[k][3])==0:
                    all_data[k][3] = clusters_t
                else:
                    for i_, cls_t_ in enumerate(clusters_t):
                        all_data[k][3][i_].extend(cls_t_)
                        
                        
                if  len(all_data[k][4])==0: 
                    all_data[k][4] = pages_t
                    
                else:
                    for i_, p_t_ in enumerate(pages_t):
                        try:
                            all_data[k][4][i_].extend(p_t_)
                        except:
                            import pdb;pdb.set_trace()
                            print('A')

        all_sampled[K] = all_samples
        all_data_cl[K] = all_data
        

    
    return all_data_cl