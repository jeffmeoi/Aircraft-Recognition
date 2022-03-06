#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, io
import numpy as np
from matplotlib import pyplot as plt
import cv2
import utils
import json


# In[2]:


img_size = 224
img_dir = utils.rotation_dir_224

def generate(annotation_file, dest_file_name):
    annotations = utils.getAnnotations(annotation_file, isXml=False)
    res = []
    for annotation in annotations:
        for i in range(2):
            img, points = utils.getImageAndPoints(annotation, utils.img_dir, img_size)
            if(i is 1):
                img=cv2.flip(img,1)
                for point in points:
                    point[0] = img_size - point[0]
            angle = 0
            while(angle < 360):
                rotatedImage, rotatedPoints = utils.rotateImageAndPoints(img, points, angle, img_size)
                """
                plt.imshow(rotatedImage)
                plt.show()
                annotatedImage = np.zeros(rotatedImage.shape[:2])
                cv2.fillPoly(annotatedImage, [rotatedPoints], 255)
                plt.imshow(annotatedImage)
                plt.show()
                """
                file_dir_path = img_dir + annotation['label'] + '/'
                if(not os.path.exists(file_dir_path)):
                    os.makedirs(file_dir_path)
                filename = img_dir + annotation['label'] + '/' + str(i) + '.' + str(angle) + '.' + annotation['name']
                ps = rotatedPoints.reshape([-1]).tolist()
                anno = {
                    'name': annotation['name'], 
                    'label': annotation['label'], 
                    'seq': annotation['seq'],
                    'task_id': annotation['task_id'], 
                    'angle': angle,
                    'points': ','.join([str(x) for x in ps]), 
                    'flip': i is 1
                }
                res.append(anno)
                cv2.imwrite(filename, rotatedImage)
                angle += 5
    f = open(utils.annotations_dir + dest_file_name + '.json', 'w')
    f.write(json.dumps(res))
    f.close()


# In[ ]:


if __name__ == '__main__':
    print('Generate Train Set')
    generate(utils.train_set, 'keypoints_train_set_224')
    print('Generate Test Set')
    generate(utils.test_set, 'keypoints_test_set_224')
    print('Done')


# In[ ]:




