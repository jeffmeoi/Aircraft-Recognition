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
def generate(annotation_file, dest_file_name):
    annotations = utils.getAnnotations(annotation_file, isXml=False)
    res = []
    for annotation in annotations:
        #print(annotation)
        img, points = utils.getImageAndPoints(annotation, utils.img_dir, img_size)
        angle = utils.getAngle(points)
        transform, keypoints = utils.centralize(img, points, 75, img_size, img_size, 3)
        transform = transform.astype(np.uint8)
        #plt.imshow(transform)
        #plt.show()
        #print(angle)
        file_dir_path = utils.regular_dir + annotation['label'] + '/'
        if(not os.path.exists(file_dir_path)):
            os.makedirs(file_dir_path)
        filename = utils.regular_dir + annotation['label'] + '/' + annotation['name']
        cv2.imwrite(filename, transform)
        ps = keypoints.reshape([-1]).tolist()
        anno = {
            'name': annotation['name'], 
            'label': annotation['label'], 
            'seq': int(annotation['seq']),
            'task_id': annotation['task_id'], 
            'angle': angle,
            'points': ','.join([str(x) for x in ps])
        }
        res.append(anno)
        #print(anno)
    f = open(utils.annotations_dir + dest_file_name + '.json', 'w')
    f.write(json.dumps(res))
    f.close()


# In[3]:


if __name__ == '__main__':
    print('Generate Train Set')
    generate(utils.train_set, 'regular_train_set')
    print('Generate Test Set')
    generate(utils.test_set, 'regular_test_set')
    print('Done')


# In[ ]:




