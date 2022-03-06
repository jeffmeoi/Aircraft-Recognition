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
threshold = 0
template_dir = utils.root_dir + r'data-set/IOU-templates/'
templates = []
for dir_name in os.listdir(template_dir):
    dir_path = os.path.join(template_dir, dir_name)
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name != '.DS_Store':
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                for i in range(len(img)):
                    for j in range(len(img[0])):
                        if img[i, j] >= 128:
                            img[i, j] = 1
                        else:
                            img[i, j] = 0
                #plt.imshow(img)
                #plt.show()
                templates.append({
                    'file_name': file_name,
                    'type': dir_name,
                    'img': img
                })


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


# In[6]:


def segmentMethod(img, anno):
    coarse = aircraft_recognition.coarseSegment(img)
    pred = coarse.copy()
    coarse = np.transpose(np.argmax(coarse, axis=1), [1, 2, 0])
    fine = aircraft_recognition.predict(img, pred=pred).astype(np.uint8)
    #plt.imshow(coarse)
    #plt.show()
    #plt.imshow(fine)
    #plt.show()
    similarity = np.sum(fine) / np.sum(coarse)
    #print('similarity', similarity)
    if similarity < threshold:
        return None
    img_k = utils.resize(img, keypoint_detect_img_size, keypoint_detect_img_size)
    angle, keypoints = keypoints_detect_network.getAngleAndKeypoints(img_k)
    keypoints = keypoints * img_size / keypoint_detect_img_size
    standard, keypoints = utils.centralize(fine, keypoints, 75, img_size, img_size, 1)
    standard = standard.astype(np.uint8)
    #plt.imshow(standard)
    #plt.show()
    probabilities = []
    for template in templates:
        #plt.imshow(template['img'])
        #plt.show()
        prob = utils.IOU(np.squeeze(standard, axis=2), template['img'])
        #print(prob)
        #print(template['img'].shape)
        probabilities.append(prob)
    return probabilities


# In[7]:


def eval():
    y_pred = []
    y_true = []
    tot_err = 0
    unqualified = 0
    for i, (img, anno) in enumerate(test_data):
        img = img.permute(1, 2, 0).detach().numpy()
        start_time = time.time()
        probabilities = segmentMethod(img, anno)
        end_time = time.time()
        print('delta', end_time - start_time)
        if probabilities is None:
            unqualified = unqualified + 1
        else:
            ind = torch.argmax(torch.tensor(probabilities)).item()
            label = templates[ind]['type']
            ground_truth = utils.getLabelBySeq(int(anno['seq']))
            print(label, ground_truth, label == ground_truth, probabilities[ind])
            y_pred.append(utils.getSeqByLabel(label))
            y_true.append(int(anno['seq']))
        
        """
        ground_truth_angle = utils.getAngle(anno['points'])
        err = abs(angle -  ground_truth_angle)
        if err > 360 - err:
            err = 360 - err
        print('angle error', err)
        tot_err = tot_err + err
        """
    print('unqualified', unqualified, 'total', len(test_data))
    print('Macro F1 Score', f1_score(y_true, y_pred, average='macro'))
    print('Weighted F1 Score', f1_score(y_true, y_pred, average='weighted'))
    print('Accuracy Score', accuracy_score(y_true, y_pred))
    print('mean error', tot_err / len(test_data))


# In[8]:


if __name__ == '__main__':
    eval()


# In[ ]:




