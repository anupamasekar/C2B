import os
import numpy
import cv2
import glob

data_path = '/media/data/prasan/C2B/anupama/dataset/Adobe_240fps/original_videos/*'
videos_list = sorted(glob.glob(data_path)) # directory in which videos are stored
save_path = '/media/data/prasan/C2B/anupama/dataset/Adobe_240fps/frames' # directory to store the extracted frames
if not os.path.exists(save_path):
    os.mkdir(save_path) # make a dire
for k,video in enumerate(videos_list):
    print('Processing video %d'%k)
    if not os.path.exists(os.path.join(save_path,'video_%.3d'%k)):
        os.mkdir(os.path.join(save_path,'video_%.3d'%k))
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    success = True
    success,image = vidcap.read()
    if image.shape[0] == 1280: 
    # some of the videos are vertical, i.e. they are of shape [1280,720]
    # you can choose to delete such videos or you can transpose those images before saving
    # I just note the video ID (k) and then delete the folder
      print('Portrait video', k)
    while success:
      success,image = vidcap.read()
      cv2.imwrite(os.path.join(save_path,'video_%.3d'%k,'frame_%.4d.png'%count), image)     # save frame as JPEG file
      count += 1