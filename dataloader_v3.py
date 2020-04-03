'''
------------------------------------------
DEFINE DATALOADER TO FETCH VIDEO SEQUENCES
------------------------------------------
removed recurrence in train data
added return of patches
replaced function to read images ad extract patches
'''
import torch
from torch.utils import data
import glob
import os
import numpy as np
import scipy.misc
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.io import imread_collection,concatenate_images
# import time

fr_per_video = 500
class Dataset_load(data.Dataset):
    
    def __init__(self, list_IDs, **input_params):
        'Initialization'

        self.list_IDs = list_IDs
        self.sub_frames = input_params['sub_frames']
        self.height_img = input_params['height']
        self.width_img = input_params['width']
        self.patch_size = input_params['patch_size']
        self.num_patches = input_params['num_patches']


    def __len__(self):
        'Denotes the total number of samples'

        num_vid = len(self.list_IDs) // fr_per_video
        num_samples = len(self.list_IDs) - num_vid*(self.sub_frames-1)
        return num_samples


    def __getitem__(self, index):
        'Generates one sample of data'

        vid_len = fr_per_video - (self.sub_frames - 1)
        vid_index = index // vid_len
        frame_index = index % vid_len
        index = vid_index*fr_per_video + frame_index
        # t1 = time.time()
        vid = concatenate_images(imread_collection(self.list_IDs[index:index+self.sub_frames]))
        vid = vid.mean(3) # conversion to grayscale; more sophisticated conversion can also be used
        vid = np.transpose(vid/255.,[1,2,0])
        # t2 = time.time()
        patches = extract_patches_2d(vid,[self.patch_size,self.patch_size],max_patches=self.num_patches)
        patches = np.transpose(patches,[0,3,1,2])
        patches = torch.FloatTensor(patches.squeeze())
        # t3 = time.time()
        # print(t2-t1, t3-t2)
        return patches #(Np,9,P,P)