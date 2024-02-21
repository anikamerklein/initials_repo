import glob
import os
import pandas as pd
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import Counter
 
from plotting.plot_corpus import plot_evolution_in_one, cmap_map
from evaluation.corpus_inference import entropy_exp
from utils import get_meta_initials, set_up_dir




H_all = pickle.load(open('/Users/amarklein/Initials_BA/initials/Reps/2dreps-14default_gram_norm.pickle', 'rb'))
M_all = pickle.load(open('/Users/amarklein/Initials_BA/initials/data/images_list.pickle', 'rb'))

all_pages, years, books, books_unique,  book_dict = get_meta_initials(M_all)


K=50 # Clustersize
seed=1
metric='euclidean'
cl_dict_dir = '/Users/amarklein/Initials_BA/initials/data/VGG16_Initials_Gram_default_clusters_10-90_28.12.pickle'
bins_list= np.arange(1494, 1647+1)

###### Temporal Evolution #######

plot_dir_ =  './Initials_repo/insights/entropy_plots/'
set_up_dir(plot_dir_)

        

t_sigma=4
n_samples=50
Ks = [K]
sample_sizes = [20, 40, 80, 150]
    
all_data_all_dim = entropy_exp(H_all, M_all, books, years, cl_dict = cl_dict_dir, Ks  = Ks, thresholds=[0], n_sims=1, t_sigma=t_sigma, n_samples=n_samples, compute_clusters=False, sample_sizes = [20, 40, 80, 150])


# Sphaera data
all_data_ = entropy_exp(H_all, M_all, books, years, cl_dict = None, Ks  = Ks, thresholds=[0], n_sims=1, t_sigma=t_sigma, n_samples=n_samples, compute_clusters=True, sample_sizes = [20, 40, 80, 150])


# Random baseline
np.random.seed(1)
H_random = np.random.normal(0,1,H_all.shape)
all_data_rand = entropy_exp(H_all, M_all, books, years, cl_dict = None, Ks  = Ks, thresholds=[0], n_sims=1, H_cluster=H_random, t_sigma=t_sigma, n_samples=n_samples, sample_sizes = [20, 40, 80, 150])




res_evolve = {'random': all_data_rand, 'sphaera': all_data_all_dim } # change here for 
pickle.dump(res_evolve, open('Insights/data_insights/evolution.pickle', 'wb'))

##### Prepare Data for Bar Plot #####

years_sorted = sorted(years)
sorted_counted = Counter(years_sorted)

range_length = list(range(min(years), max(years))) # Get the largest value to get the range.
data_series = {}

for i in range_length:
    data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.

for key, value in sorted_counted.items():
    data_series[key] = value

data_series = pd.Series(data_series)
x_values = data_series.index

    
####### 2. Plot Temporal Evolution #######
    
res_ = pickle.load( open('Insights/data_insights/evolution.p', 'rb'))
all_data_rand = res_['random']
all_data = res_['sphaera']

keys_ = [k[0] for k in all_data[K].keys()]


cmap_bone  = plt.get_cmap('bone_r')
cmap_bone = cmap_map(lambda x: -0.05+x*0.95,cmap_bone)
cdict_random = {0:cmap_bone(0)}
cdict_random = {sample:cmap_bone(sample) for sample in sample_sizes}

f=plt.figure( figsize=(6,3), dpi=200)
ax = plt.subplot(1, 1, 1)

ax2 = ax.twinx()

cmap  = plt.get_cmap('cool')
cmap = cmap_map(lambda x: x*0.75,cmap)
cdict = {sample:cmap(nb*200) for nb, sample in enumerate(sample_sizes)}
legs_,_ = plot_evolution_in_one(all_data_rand[K], fax=(f,ax), cdict=cdict_random)
legs_,_ = plot_evolution_in_one(all_data[K], fax=(f,ax), cdict=cdict, legs=legs_)

ax2.bar(x_values, data_series.values)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('timestep t')
ax.set_ylabel('entropy')
ax2.set_ylabel('nb of initials per year')

plt.legend( list(zip(legs_[len(keys_):],legs_[:len(keys_)])), 
       [str(k_) for k_ in keys_],
      handler_map={tuple: HandlerTuple(ndivide=None)},
       bbox_to_anchor=(0.985, 0.54), borderpad=0.2,  prop={'size': 11}) 
f.tight_layout()
f.savefig(os.path.join(plot_dir_, 'evolution_clnb.png'), dpi=300, transparent=False)
plt.show()