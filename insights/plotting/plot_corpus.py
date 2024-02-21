import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
import palettable
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy
import itertools


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)
       

    

def plot_evolution_in_one(all_data, fax=None, kwargs={}, keys = None, cdict={}, legs=[], color=None, ls='-'):
    if fax:
        f,axs = fax
    else:
        raise

    i=0
    if keys is None:
        keys = list(all_data.keys())
        
    plot_legs = True if len(legs)>0 else False
        
    entropies = {}
    for k in keys:
        v = all_data[k]
        thresh = k[0]
        temps = v[0]
        e = np.nanmean(v[1],axis=0)
        counts = np.nanmean(v[2],axis=0)
        c_ = color if color is not None else cdict[k[0]]
        l, = axs.plot(temps, e, c=c_, label=str(k[1]), linestyle=ls)

        entropies[k[0]] = e 
        legs.append(l)

    if fax is None:
        plt.show()

    return legs, entropies



def plot_bar(x, labels, n_tables, fax=None, sort='alphabetical'):

    if fax:
        f,ax = fax
    else:
        f, ax = plt.subplots(1,1, figsize=(6,3))

        
    #labels= shorten(labels)
    
    if sort=='alphabetical':
        inds_ = np.argsort(labels)

        
    elif sort == 'value':
        inds_ = np.argsort(x)

    else:
        raise
        
    
    labels_ = np.array(labels)[inds_]
    x_ = x[inds_]  
    n_tables_ = n_tables[inds_]


    c_bars =  np.copy(np.array(['#0000FF']*(len(labels))))
    c_bars[n_tables_<=100]= '#0a0a0d'
    c_bars[n_tables_<=50]= '#d5d5d5'
    print(c_bars)
        
    ax.bar(labels_, x_, align='center', color=c_bars , width=0.75)

    ax.tick_params(axis='x', which='major',  rotation=90)
    ax.tick_params(axis='x', which='minor', rotation=90)
    


    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels = labels_, rotation=90,  ha="center")

    
    ax.set_yticks(np.arange(0.0, 4.0, 1.))
    ax.set_yticklabels(labels = ['{:0.1f}'.format(x) for x in np.arange(0.0, 4.0, 1.)]) #, rotation=90,  ha="center")

    
    ax.set_ylabel(r'$H(p)$', fontsize=17) 
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    

    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    
    
    if fax is None:
        plt.show()
        
