
from PIL import Image
import cv2
import numpy as np
import PIL.ImageOps
import matplotlib.pyplot as plt
import pickle

def make_contact_sheet(fnames,ncols,nrows,photow,photoh, marl,mart,marr,marb, padding, path, binarize = False):
    """\
    Make a contact sheet from a group of filenames:

    fnames       A list of names of the image files
    
    ncols        Number of columns in the contact sheet
    nrows        Number of rows in the contact sheet
    photow       The width of the photo thumbs in pixels
    photoh       The height of the photo thumbs in pixels

    marl         The left margin in pixels
    mart         The top margin in pixels
    marr         The right margin in pixels
    marl         The left margin in pixels

    padding      The padding between images in pixels

    returns a PIL image object.
    """

    # Read in all images and resize appropriately
    imgs = [Image.open(path + fn).resize((photow,photoh)) for fn in fnames]
    
    if binarize:
        #convert from PIL zu CV2
        imgs = [PIL.ImageOps.grayscale(i) for i in imgs]
        imgs = [np.array(i) for i in imgs]
        imgs = [binarize_imagedata(i) for i in imgs]

    # Calculate the size of the output image, based on the
    #  photo thumb sizes, margins, and padding
    marw = marl+marr
    marh = mart+ marb

    padw = (ncols-1)*padding
    padh = (nrows-1)*padding
    isize = (ncols*photow+marw+padw,nrows*photoh+marh+padh)

    # Create the new image. The background doesn't have to be white
    white = (255,255,255)
    inew = Image.new('RGB',isize,white)

    # Insert each thumb:
    for irow in range(nrows):
        for icol in range(ncols):
            left = marl + icol*(photow+padding)
            right = left + photow
            upper = mart + irow*(photoh+padding)
            lower = upper + photoh
            bbox = (left,upper,right,lower)
            try:
                img = imgs.pop(0)
            except:
                break
            inew.paste(img,bbox)
    return inew



def binarize_imagedata(img):
        image = cv2.GaussianBlur(img,(5,5),0)
        img = image.astype(np.uint8)
        ret, binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary3 = np.zeros((binary.shape[0], binary.shape[1], 3))
        binary3[:,:,0] = binary
        binary3[:,:,1] = binary
        binary3[:,:,2] = binary

        return Image.fromarray(binary3.astype(np.uint8))


        # dann noch normalisieren

# noch umbenennen! Kann nur von classlists benutzt werden, extra funktion für cluster

def vis_data_by_label(representation, class_lists, class_names, save_name, title, plot_line = False, intercept = None, slope = None ): # class names includes None

    ims = pickle.load(open('./data/images_list.pickle', 'rb'))
    labels = pickle.load(open('./data/lettersdf.pickle', 'rb'))
    data = pickle.load(open(representation, 'rb'))

    c_labels = ([None] * len(ims)).append([None])
    labels["cluster_labels"] = c_labels
    c_numbers =([0] * (len(ims)+1))
    
    labels["labels_binary"] = c_numbers

    for row in labels.iterrows():
        idx = 1
        for cl in class_lists: 
            if row[1]["initial"] in cl:
                labels.loc[row[0], "cluster_labels"] = class_names[idx]
                labels.loc[row[0], "labels_binary"] = idx
            idx +=1

    labels_binary = [list(labels[labels.initial == im].labels_binary)[0] for im in ims]



    fig = plt.figure(figsize= (25,25))
    ax = fig.add_subplot()
    sc = plt.scatter(data[:,0], data[:,1], c = labels_binary, alpha=0.5, cmap = "tab20_r")
    if plot_line == True:
        start = slope*min(data[:,0])+intercept
        end = slope*max(data[:,0])+intercept
        plt.plot([min(data[:,0]), start], [max(data[:,0]), end], 'k-', color = 'r')


    plt.legend(handles=sc.legend_elements()[0], labels=class_names, title = title) # maybe take title out again
    plt.show()
    plt.savefig(save_name+ '.pdf')


def vis_data_by_cluster(representation, cluster_list, class_names, save_name, title, plot_line = False, intercept = None, slope = None ): # class names includes None

    #ims = pickle.load(open('./data/images_list.pickle', 'rb'))
    #labels = pickle.load(open('./data/lettersdf.pickle', 'rb'))
    data = pickle.load(open(representation, 'rb'))

    labels_binary = cluster_list
    



    fig = plt.figure(figsize= (25,25))
    ax = fig.add_subplot()
    sc = plt.scatter(data[:,0], data[:,1], c = labels_binary, alpha=0.5, cmap = "tab20_r")
    if plot_line == True:
        start = slope*min(data[:,0])+intercept
        end = slope*max(data[:,0])+intercept
        plt.plot([min(data[:,0]), start], [max(data[:,0]), end], 'k-', color = 'r')


    plt.legend() # maybe take title out again
    plt.show()
    plt.savefig(save_name+ '.pdf')
