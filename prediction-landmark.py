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
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import pydensecrf.densecrf as dcrf
import utils
import json
import keypoints_detect_network_224 as keypoints_detect_network
import aircraft_recognition
from sklearn.metrics import f1_score, accuracy_score
import time

# In[2]:


img_size = 224
keypoint_detect_img_size = 224
templates = utils.getAnnotations(os.path.join(utils.root_dir, r'data-set/landmark-templates.json'), isXml=False)
for template in templates:
    template['points'] = np.array([float(x) for x in template['points'].split(',')]).reshape(-1, 2)


# In[3]:


def getImageAndPoints(annotation, img_dir, img_size=224):
    img_path = img_dir + annotation['label'] + '/' + annotation['name']
    img = cv2.imread(img_path)
    if(img is None):
        print(annotation, img_path)
    annotation['points'] = (np.array(annotation['points'])*img_size/max(img.shape[0], img.shape[1])).astype(np.float32)
    return utils.resize(img, img_size, img_size), annotation['points']


# In[4]:


class TypeDataset(Dataset):
    def __init__(self, annotations_file=utils.annotations_dir + 'train_set.json', 
                 img_dir=utils.img_dir):
        regular_file = open(annotations_file, 'r')
        regular_str = regular_file.read()
        annotations = json.loads(regular_str)
        regular_file.close()
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img, points = getImageAndPoints(self.annotations[idx], self.img_dir, img_size)
        img = torch.from_numpy(img).float().permute(2,0,1)
        return img, self.annotations[idx]


# In[5]:


train_data = TypeDataset(annotations_file=utils.annotations_dir + 'train_set.json')
test_data = TypeDataset(annotations_file=utils.annotations_dir + 'test_set.json')


# In[9]:


def landmarkMethod(img):
    img_k = utils.resize(img, keypoint_detect_img_size, keypoint_detect_img_size)
    angle, keypoints = keypoints_detect_network.getAngleAndKeypoints(img_k)
    keypoints = keypoints * img_size / keypoint_detect_img_size
    standard, keypoints = utils.centralize(img, keypoints, 75, img_size, img_size, 3)
    #cv2.polylines(standard, [keypoints], True, (255, 0, 0))
    #plt.imshow(standard.astype(np.uint8))
    #plt.show()
    distances = []
    for template in templates:
        distances.append(np.sum((keypoints-template['points'])**2))
    distances = np.array(distances)
    ind = np.argmin(distances)
    return templates[ind]['seq']


# In[10]:


def eval():
    y_pred = []
    y_true = []
    for i, (img, anno) in enumerate(test_data):
        img = img.permute(1, 2, 0).detach().numpy()
        start_time = time.time()
        pred = landmarkMethod(img)
        end_time = time.time()
        delta_time = end_time - start_time
        print('delta', delta_time)
        print(utils.getLabelBySeq(pred), utils.getLabelBySeq(int(anno['seq'])), pred == int(anno['seq']))
        y_pred.append(pred)
        y_true.append(anno['seq'])
    print('Macro F1 Score', f1_score(y_true, y_pred, average='macro'))
    print('Weighted F1 Score', f1_score(y_true, y_pred, average='weighted'))
    print('Accuracy Score', accuracy_score(y_true, y_pred))


# In[11]:


if __name__ == '__main__':
    eval()


# In[ ]:




