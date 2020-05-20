'''
-----------------------------------
DEFINE C2B SENSOR FOR BUCKET IMAGES
-----------------------------------
'''
import torch
import torch.nn as nn
import numpy as np
import scipy.io


class C2B(nn.Module):
    
    def __init__(self, block_size, sub_frames, mask='random', two_bucket=False):
        super(C2B, self).__init__()

        if mask == 'impulse':
            assert block_size**2 == sub_frames
            code = torch.eye(block_size**2).cuda()
            code = code.reshape(1, sub_frames, block_size, block_size)
            print('Initialized sensor with impulse code %dx%dx%d'%(sub_frames, block_size, block_sizes))
        
        elif mask == 'opt':
            ## eccv code 16x8x8
            filename = '/data/prasan/anupama/dataset/eccv18/optimized_SBE_8x8x16'
            code = scipy.io.loadmat(filename)['x']
            # code = code[::2, ::2, :]
            code = code.transpose(2,0,1)
            assert code.shape == (16,8,8)
            code = torch.cuda.FloatTensor(code).unsqueeze(0)   
            print('Initialized sensor with optimized code from %s'%filename)
        
        else:
            ## random code 16x8x8
            code = torch.empty(1, sub_frames, block_size, block_size).cuda()
            code = code.bernoulli_(p=0.2)
            print('Initialized sensor with random code %dx%dx%d'%(sub_frames, block_size, block_size))
        
        self.block_size = block_size
        if mask == 'train':
            self.trainable = True
        self.code = nn.Parameter(code, requires_grad=self.trainable)
        self.two_bucket = two_bucket
        # self.code_repeat = code.repeat(1, 1, patch_size//block_size, patch_size//block_size)


    def update_mask(self):
        assert self.trainable
        code = torch.round(self.code.data)
        self.code.data.copy_(code)


    def forward(self, x):
        _,_,H,W = x.size()
        code_repeat = self.code.repeat(1, 1, H//self.block_size, W//self.block_size)
        b1 = torch.sum(code_repeat*x, dim=1, keepdim=True) / torch.sum(code_repeat, dim=1, keepdim=True)
        if not self.two_bucket:
            return b1
        code_repeat_comp = 1 - code_repeat
        b0 = torch.sum(code_repeat_comp*x, dim=1, keepdim=True) / torch.sum(code_repeat_comp, dim=1, keepdim=True)
        # blurred = torch.mean(x, dim=1, keepdim=True)
        return b1, b0 # (N,1,H,W)