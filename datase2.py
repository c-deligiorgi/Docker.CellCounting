import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
#from torchvision.transforms import Grayscalei
import pandas as pd
import pdb

class CellsDataset(Dataset):
    # a very simple dataset

    def __init__(self, root_dir, transform=None, return_filenames=False):
        self.root = root_dir
        self.transform = transform
        self.return_filenames = return_filenames
        self.files = [os.path.join(self.root,filename) for filename in os.listdir(self.root)]
        self.files = [path for path in self.files
                      if os.path.isfile(path) and os.path.splitext(path)[1]=='.png']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sample = Image.open(path)
       # transform3 = Grayscale(num_output_channels=3)
       # sample = transform3(sample) # convert to a 3 channel grayscale, as it needs to be 3 channel.
        if self.transform:
            sample = self.transform(sample)

        if self.return_filenames:
            return sample, path
        else:
            return sample

    def duplicate_and_transform(self,transforms):
        currfilelen = self.__len__()
        
        for pind in range(0,currfilelen-1):
            path = self.files[pind]
            sample = Image.open(path)
            samplet = transforms(sample)
            # save the transformed images and save the associated counts:
            newpath = path[:-4]+'trans.png' #path[:-4] is the path without the .png extension.
            samplet.save(newpath,"PNG")
            metadata = pd.read_csv(os.path.join(self.root,'gt.csv'))
            metadata.set_index('filename',inplace=True)
            count = metadata['count'][os.path.split(path)[-1]]
            samplet.save(newpath,"PNG")
            f = open(self.root+'gt.csv','a')
            basepath = os.path.basename(path) # strip away the directory list and just get the filename
            f.write(basepath[:-4]+'trans.png,'+str(count)+'\n')
            # return the new file names and add to self.files.
            self.files.append(newpath)
            #pdb.set_trace()
