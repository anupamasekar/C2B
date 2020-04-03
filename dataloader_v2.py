'''
------------------------------------------
DEFINE DATALOADER TO FETCH VIDEO SEQUENCES
------------------------------------------
removed recurrence in train data
added return of patches
'''
import torch
from torch.utils import data
import glob
import os
import numpy as np
import scipy.misc
from PIL import Image
# import time


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

        num_vid = len(self.list_IDs) // 500
        num_samples = len(self.list_IDs) - num_vid*(self.sub_frames-1)
        return num_samples


    def __getitem__(self, index):
        'Generates one sample of data'

        vid_len = 500 - (self.sub_frames - 1)
        vid_index = index // vid_len
        frame_index = index % vid_len
        index = vid_index*500 + frame_index
        
        # t1 = time.time()
        images = []
        for i in range(index, index+self.sub_frames):
            img = Image.open(self.list_IDs[i]).convert(mode='L')
            # img = img.resize((self.width_img, self.height_img))
            img = np.array(img) / 255.
            images.append(img) # (H,W)
        # t2 = time.time()

        vid = np.stack(images, axis=0) # (9,H,W)
        vid = torch.FloatTensor(vid)
        # t3 = time.time()

        # random patches
        pi = torch.randint(0, vid.shape[1]-self.patch_size+1, size=(self.num_patches,))
        pj = torch.randint(0, vid.shape[2]-self.patch_size+1, size=(self.num_patches,))
        patches = [vid[:, pi[i]:pi[i]+self.patch_size, pj[i]:pj[i]+self.patch_size] for i in range(self.num_patches)]
        patches = torch.stack(patches, dim=0) # (np,9,ps,ps)
        # print('patched_vid', patched_vid.shape)
        # t4 = time.time()
        # print(t2-t1, t3-t2, t4-t3)
        # patched = vid.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) 
        # (9,3,5,240,240)
        # patched = patched.view()
        # patched_vid = patched_vid.transpose(0,2)
        # # print(patched_vid.shape)
        # patched_vid = patched_vid.reshape(-1, self.sub_frames, self.patch_size, self.patch_size) 
        return patches #(Np,9,P,P)