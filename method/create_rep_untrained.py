import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mlp

import torchvision.models as models
import torch.nn as nn
import torch
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import io


class CustomDataset(Dataset):
    def __init__(self, file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            split (string): denoting the split of the data set. (train or test)
        """
        list = open(file, 'rb')
        self.images_list = pickle.load(list)
        self.root_dir = root_dir
        self.train_data = [os.path.join(self.root_dir,c) for c in self.images_list]


        self.transform = transforms.Compose([transforms.Resize((224,224))
					    ,transforms.ToTensor()
					    ,transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

        
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir,
                                self.images_list[idx])

        image = cv2.imread(img_name,0)
        image = cv2.GaussianBlur(image,(5,5),0)
        img = image.astype(np.uint8)
        ret, binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary3 = np.zeros((binary.shape[0], binary.shape[1], 3))
        binary3[:,:,0] = binary
        binary3[:,:,1] = binary
        binary3[:,:,2] = binary

        image = self.transform(Image.fromarray(binary3.astype(np.uint8)))

        return image

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)



path = r"/Users/amarklein/Downloads/initials_new" # path to images
fileending = ".jpg"

images = [] # read in images for having the bids in the right order as in the output pickle file

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith(fileending):
            images.append(file.name)
            



with open('images_list.pickle', 'wb') as f: # save images_list as pickle file for always having the same order
     pickle.dump(images, f)

transformed_images = CustomDataset("images_list.pickle" ,path)

transformed_images # cant really be saved in a pickle file...

transformed_list = [torch.unsqueeze(i, dim= 0) for i in transformed_images]

concatenated_tensor = torch.cat(transformed_list, dim=0)

cut_layer = 0

path = r"/Users/amarklein/Downloads/initals/" 

model = models.vgg16(pretrained=True)

print("step 1")

new_classifier = nn.Sequential(*list(model.classifier.children())[:-7])
model.classifier = new_classifier

# also put out avgpool? -> Oliver fragen.... brings ecerything to the shape of 7*7*x
new_avgpool = nn.Sequential(*list(model.avgpool.children())[:-1])
model.avgpool = new_avgpool

if cut_layer != 0:
    new_features = nn.Sequential(*list(model.features.children())[:cut_layer])
    model.features = new_features


outputs = []

n = 100
a = torch.split(concatenated_tensor, n)

outp= []
for count, x in enumerate(a):
    print("round "+str(count)+" of "+ int(str(len(images)/100)))
    out = model(x) #shape of [1, 4096]
    out = out.detach().numpy()
    print(out.shape)
    outp.append(out)

outcat = torch.cat([torch.from_numpy(i) for i in outp])


with open("VGG16_Initials_"+str(cut_layer)+ "untrained_binary.pickle", "wb") as f:
    pickle.dump(outcat, f)
