import glob
import os
import pandas as pd
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import json
from utils import set_up_dir, get_hist, get_meta_initials
from plotting.plot_corpus import plot_bar



H_all = pickle.load(open('/Users/amarklein/Initials_BA/initials/Reps/2dreps-14default_gram_norm.pickle', 'rb'))
M_all = pickle.load(open('/Users/amarklein/Initials_BA/initials/data/images_list.pickle', 'rb'))
all_books = pd.read_json(open('/Users/amarklein/Initials_BA/initials/data/all_books.json', 'rb'))
uniqueprinters = all_books.drop_duplicates(subset=['printers'])
uniquebooks = all_books.drop_duplicates(subset=['bid'])


all_pages, years, books, books_unique,  book_dict = get_meta_initials(M_all)
H_norm = np.sqrt(H_all)

K=1500
seed=1
metric='euclidean'
cl_dict_dir = '/Users/amarklein/Initials_BA/initials/Initials_repo/data/VGG16_Initials_Gram-14default_cluster_alldim40.pickle'

bins_list= np.arange(1494, 1647+1)
cl_dict = pickle.load(open(cl_dict_dir, 'rb'))
y_pred,_  = cl_dict 


############

plot_dir_ =  './Initials_repo/insights/entropy_plots/'
set_up_dir(plot_dir_)
        


CITIES_ENTROPY = True
PRINTERS_ENTROPY = False


if PRINTERS_ENTROPY: # only evaluate the ones that have min 25 samples 

    df_allbooks = pd.read_json('/Users/amarklein/Initials_BA/initials/initials_repo/data/all_books.json')
    sacrobosco_ids = list(df_allbooks.bid)
    sacrobosco_printers = list(set(df_allbooks.printers))


    df = pd.read_csv('/Users/amarklein/Initials_BA/initials/initials_repo/data/2dimreps_with_id_year_place_printer_proto.csv') 

    page_id = [os.path.split(m)[1] for m in M_all]
    page_id = [(p_.replace('-', '_')) for p_ in page_id]
    book_id = [int(p_.split('_')[0]) for p_ in page_id]

    printers = []

    for m in M_all:
        pid = os.path.split(m)[1] 
        bid = int(pid.split('_')[0])
        cs = df[df['id']==bid].printer.unique()
        assert len(cs) == 1
        printers.append(cs[0])# create cities list with city name for every page

    printers_map = {c:i  for i,c in enumerate(list(set(printers)))} # give every printer a number
    printers_map_inv = {v:k for k,v in printers_map.items()} # and the other way round, to be able to search for the number of a printer
    printers_num = np.array([printers_map[c] for c in printers]) # fill 
    N_printers = len(printers_map) # number of different printers

    entropy = {}

    for c in list(set(printers_num)): # set of all the numbers of printers

        printers_num_ = np.copy(printers_num)

        unique_clusters_per_printer = len(list(set(y_pred[printers_num_==c]))) # number of clusters per printer
        tables_per_printer = len(list(y_pred[printers_num_==c]))
        
        if tables_per_printer >= 44:

            books_per_printer = list(set(books[printers_num_==c])) # different books per printer


            # Compute bag of clusters 
            hh = get_hist(y_pred[printers_num_==c], K=len(list(set(y_pred)))) #histogramm of clusters by printer

            e_difference = scipy.stats.entropy(hh)
            entropy[c] = (c,  hh, 0, e_difference, unique_clusters_per_printer, tables_per_printer, books_per_printer)


    printers_= list(entropy.keys())

    labels = [printers_map_inv[c] for c in printers_]
    labels = np.array([ l + ' *' if l in sacrobosco_printers else l for l in labels])

    es = np.array([entropy[c][3] for c in printers_])
    n_tables_per_printer = np.array([entropy[c][5] for c in printers_])
    idx_ = np.array(range(len(es)))

    y =  es[idx_]/np.log(2)

    f, ax = plt.subplots(1,1, figsize=(6.,3.5))

    plot_bar(y, labels, n_tables_per_printer, fax=(f,ax))
   
    f.tight_layout()
    f.savefig(os.path.join(plot_dir_, 'printer_entropy_bar.png'), dpi=300, transparent=False)
    plt.close()
            
if CITIES_ENTROPY:

    df_sacrobosco = pd.read_json('data/all_books.json')
    sacrobosco_ids = list(df_sacrobosco.bid)
    sacrobosco_cities = list(set(df_sacrobosco.place))


    df = pd.read_csv('/Users/amarklein/Initials_BA/initials/data/2dimreps_with_id_year_place_printer_proto.csv')

    page_id = [os.path.split(m)[1] for m in M_all]
    page_id = [(p_.replace('-', '_')) for p_ in page_id]
    book_id = [int(p_.split('_')[0]) for p_ in page_id]

    cities = []

    for m in M_all:
        pid = os.path.split(m)[1] 
        bid = int(pid.split('_')[0])
        cs = df[df['id']==bid].place.unique()
        assert len(cs) == 1
        cities.append(cs[0]) # create cities list with city name for every page

    cities_map = {c:i  for i,c in enumerate(list(set(cities)))} # give every city a number
    cities_map_inv = {v:k for k,v in cities_map.items()} # and the other way round, to be able to search for the number of a city
    cities_num = np.array([cities_map[c] for c in cities]) # fill 
    N_cities = len(cities_map) # number of different cities

    entropy = {}

    for c in list(set(cities_num)): # set of all the numbers of cities == np.arange(N_cities)

        cities_num_ = np.copy(cities_num)

        unique_clusters_per_city = len(list(set(y_pred[cities_num_==c]))) # number of clusters per city
        tables_per_city = len(list(y_pred[cities_num_==c]))

        books_per_city = list(set(books[cities_num_==c])) # different books per city

        # Compute bag of clusters 
        hh = get_hist(y_pred[cities_num_==c], K=len(list(set(y_pred)))) #histogramm of clusters in a city
      


        e_difference = scipy.stats.entropy(hh)
        entropy[c] = (c,  hh, 0, e_difference, unique_clusters_per_city, tables_per_city, books_per_city) 



    cities_= np.array(range(N_cities))

    labels = [cities_map_inv[c] for c in cities_]
    labels = np.array([ l + ' *' if l in sacrobosco_cities else l for l in labels])
    es = np.array([entropy[c][3] for c in cities_])
    n_tables_per_city = np.array([entropy[c][5] for c in cities_])
    idx_ = np.array(range(len(es)))

    y =  es[idx_]/np.log(2) 

    f, ax = plt.subplots(1,1, figsize=(6.,3.5))
    plot_bar(y, labels, n_tables_per_city, fax=(f,ax)) 
   
    f.tight_layout()
    f.savefig(os.path.join(plot_dir_, 'location_entropy_40.png'), dpi=300, transparent=False)
    plt.close()






