import h5py
import os
import glob
import torch
import numpy as np
from PIL import Image
import utils

data_path = '/media/data/prasan/datasets'
save_path = '/media/data/prasan/C2B/anupama/dataset/GoPro_patches_ds2_s16-8_p128-64.hdf5'

patch_size = 128
subframes = 16
frames_per_seq = 512 
downsample = 2

test_path = sorted(glob.glob(data_path + '/test_set/GOPR*/'))
test_images = []
for path in test_path:
    test_images.extend(sorted(glob.glob(path + '*.png'))[:frames_per_seq]) 

test_folders = [f[-15:-1] for f in test_path]

training_path = sorted(glob.glob(data_path + '/GOPR*/'))
training_path = [f for f in training_path if f[-15:-1] not in test_folders]
training_images = []
for path in training_path:
    training_images.extend(sorted(glob.glob(path + '*.png'))[:frames_per_seq])

print('Training paths: %d'%(len(training_images)))
print('Test paths: %d'%(len(test_images)))



## train data
for i in range(len(training_images)//frames_per_seq):
    for j in range(frames_per_seq):
        print('Train Loop:', i, '/', j)
        img = Image.open(training_images[i*frames_per_seq+j]).convert(mode='L')
        img = np.array(img)
        img = img[np.newaxis, ::downsample, ::downsample]
        if j == 0:
            data_arr = img
        else:
            data_arr = np.concatenate([data_arr, img], axis=0)
    print(data_arr.shape)
    data_tensor = torch.FloatTensor(data_arr)
    data_tensor = data_tensor.unfold(0, subframes, subframes//2)\
                    .unfold(1, patch_size, patch_size//2).unfold(2, patch_size, patch_size//2)
    print(data_tensor.shape)
    data_tensor = data_tensor.reshape(-1, subframes, patch_size, patch_size)
    print(data_tensor.shape)
    if i == 0:
        train_tensor = data_tensor
    else:
        train_tensor = torch.cat([train_tensor, data_tensor], dim=0)
    # if i == 0:
    #     break

print('Read all train images', train_tensor.shape)
train_arr = train_tensor.data.numpy()

f = h5py.File(save_path, 'w')
f.create_dataset('train', dtype='uint8', data=train_arr)
print('Created train dataset')


# ## test data
# for i in range(len(test_images)//frames_per_seq):
#     for j in range(frames_per_seq):
#         print('Test Loop:', i, '/', j)
#         img = Image.open(test_images[i*frames_per_seq+j]).convert(mode='L')
#         img = np.array(img)
#         img = img[np.newaxis, ::downsample, ::downsample]
#         if j == 0:
#             data_arr = img
#         else:
#             data_arr = np.concatenate([data_arr, img], axis=0)
#     print(data_arr.shape)
#     data_tensor = torch.FloatTensor(data_arr)
#     data_tensor = data_tensor.unfold(0, subframes, subframes//2)\
#                     .unfold(1, patch_size, patch_size//2).unfold(2, patch_size, patch_size//2)
#     print(data_tensor.shape)
#     data_tensor = data_tensor.reshape(-1, subframes, patch_size, patch_size)
#     print(data_tensor.shape)
#     if i == 0:
#         test_tensor = data_tensor
#     else:
#         test_tensor = torch.cat([test_tensor, data_tensor], dim=0)
#     # if i == 0:
#     #     break

# print('Read all test images', test_tensor.shape)
# test_arr = test_tensor.data.numpy()

# # f = h5py.File(save_path, 'w')
# f.create_dataset('test', dtype='uint8', data=test_arr)
# print('Created test dataset')
# print('Saved datasets to %s'%(save_path))



## reading h5py file
# f = h5py.File(save_path, 'r')
# print(list(f.keys()))
# train_dset = f['train']
# print('train', train_dset.dtype, train_dset.shape)
# test_dset = f['test']
# print('test', test_dset.dtype, test_dset.shape)
# test_vid = train_dset[0, ...]
# utils.save_gif(test_vid/255., 'test.gif')