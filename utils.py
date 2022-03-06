#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, io
import numpy as np
from matplotlib import pyplot as plt
import cv2
import xml.sax
import math
from functools import reduce
from sklearn.cluster import KMeans
import json


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using {} device'.format(device))# hyper parameters

root_dir = r'/home/tzt/jeffxie/'
img_dir = root_dir + r'data-set/airplane-dataset-trans/'
crf_dir = root_dir + r'data-set/airplane-dataset-trans-crf/'
rotation_dir = root_dir + r'data-set/airplane-dataset-trans-rotations/'
rotation_dir_80 = root_dir + r'data-set/airplane-dataset-trans-rotations-80/'
rotation_dir_224 = root_dir + r'data-set/airplane-dataset-trans-rotations-224/'
regular_dir = root_dir + r'data-set/airplane-dataset-trans-regular/'
annotations_dir = root_dir + r'data-set/'
train_set = annotations_dir + r'train_set.json'
test_set = annotations_dir + r'test_set.json'


# In[3]:


def resize(img, dest_height=224, dest_width=224):
    height, width = img.shape[:2]
    right = bottom = 0
    if(height > width):
        right = height - width
    else:
        bottom = width - height
    img_square = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, None, [0,0,0])
    res = cv2.resize(img_square, (dest_height,dest_width))
    return res


# In[4]:


def getImageAndPoints(annotation, img_dir=img_dir, img_size=224):
    img_path = img_dir + annotation['label'] + '/' + annotation['name']
    img = cv2.imread(img_path)
    if(img is None):
        print(annotation, img_path)
    points = (np.array(annotation['points'])*img_size/max(img.shape[0], img.shape[1])).astype(np.float32)
    img = resize(img, img_size, img_size)
    return img, points


# In[5]:


def rotatePoints(ps, m):
    if(ps is None):
        return
    pts = np.float32(ps)#.reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))]
    return np.array(target_point).astype(np.int32)

def rotateImageAndPoints(img, points, angle, img_size, resize_rate=1):
    M = cv2.getRotationMatrix2D((img_size//2,img_size//2), angle, resize_rate)
    res_img = None
    if img is not None:
        h,w,c = img.shape
        res_img = cv2.warpAffine(img, M, (w, h))
    out_points = rotatePoints(points,M)
    return res_img, out_points


# In[6]:


def str2position(str):
    list = str.split(',');
    pos = [round(float(list[0])), round(float(list[1]))]
    return pos
class XmlHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.images = []
        self.current_image = ''
        self.labels = {
            '147560': 'Boeing',
            '147655': 'B-1',
            '148026': 'B-2',
            '153105': 'A-10',
            '153163': 'A-26',
            '153203': 'B-29',
            '154799': 'B-52',
            '156147': 'C-5',
            '156184': 'C-17',
            '156667': 'C-21',
            '157571': 'C-130',
            '166985': 'C-135',
            '167423': 'E-3',
            '167479': 'F-16',
            '167582': 'F-22',
            '167945': 'KC-10',
            '167946': 'P-63',
            '167926': 'T-6',
            '167947': 't-43',
            '167948': 'U-2'

        }
        self.seqs = {
            '147560': 0,
            '147655': 1,
            '148026': 2,
            '153105': 3,
            '153163': 4,
            '153203': 5,
            '154799': 6,
            '156147': 7,
            '156184': 8,
            '156667': 9,
            '157571': 10,
            '166985': 11,
            '167423': 12,
            '167479': 13,
            '167582': 14,
            '167945': 15,
            '167946': 16,
            '167926': 17,
            '167947': 18,
            '167948': 19
        }
    
    # 元素开始事件处理
    def startElement(self, tag, attributes):
        if(tag == 'image'):
            self.current_image = {
                'name': attributes['name'],
                'seq': self.seqs[attributes['task_id']], 
                'label': self.labels[attributes['task_id']], 
                'task_id': attributes['task_id'],
                'points': None
            }
        elif(tag == 'points'):
            self.current_image['points'] = list(map(str2position, attributes['points'].split(';')))
            self.images.append(self.current_image)
            
    # 元素结束事件处理
    def endElement(self, tag):
        if(tag == 'image'):
            current_image = None
    
    # 内容事件处理
    def characters(self, content):
        pass
def getAnnotations(url=annotations_dir + "annotations.xml", isXml=True):
    if isXml:
        # 创建一个 XMLReader
        parser = xml.sax.make_parser()
        # turn off namepsaces
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)

        # 重写 ContextHandler
        Handler = XmlHandler()
        parser.setContentHandler(Handler)
        parser.parse(url)
        return Handler.images
    else:
        file = open(url, 'r')
        strs = file.read()
        annotations = json.loads(strs)
        file.close()
        return annotations


# In[7]:


if __name__ == '__main__':
    annotations = getAnnotations()
    print(len(annotations))


# In[8]:


class MinCircle:
    def __init__(self, center, radius, points, inCircumference, isCircumcircle):
        self.center = center
        self.radius = radius
        self.points = points
        self.inCircumference = inCircumference
        self.isCircumcircle = isCircumcircle
    def inCircle(self, point):
        return Point.dist(point, self.center) <= self.radius
    def __str__(self):
        return "center: {}, radius: {}, points: {}, inCircumference: {}, isCircumcircle: {}".format(self.center, 
            self.radius, self.points, self.inCircumference, self.isCircumcircle)
    def __repr__(self):
        return str(self)
    def getPointNotInOriginPoints(self, points):
        for point in points:
            if point not in self.points:
                return point
        return None


# In[9]:


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "({},{})".format(self.x, self.y)
    def __repr__(self):
        return str(self)
    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)
    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)
    def dot(self, p):
        return self.x*p.x + self.y*p.y
    @staticmethod
    def dist(a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
    @staticmethod
    def mid(a, b):
        return Point((a.x+b.x)/2, (a.y+b.y)/2)
    def __eq__(self, p):
        return self.x == p.x and self.y == p.y
    def __hash__(self):
        return int(self.x)
    @staticmethod
    def circumcircle(A, B, C):#外接圆
        a = A.x - B.x
        b = A.y - B.y
        c = A.x - C.x
        d = A.y - C.y
        a1 = ((A.x * A.x - B.x * B.x) + (A.y * A.y - B.y * B.y)) / 2.0
        a2 = ((A.x * A.x - C.x * C.x) + (A.y * A.y - C.y * C.y)) / 2.0
        theta = b * c - a * d
        if abs(theta) < 1e-7:
            raise RuntimeError('There should be three different x & y !')
        x0 = (b * a2 - d * a1) / theta
        y0 = (c * a1 - a * a2) / theta
        r = math.sqrt(pow((A.x - x0), 2) + pow((A.y - y0), 2))
        return Point(x0, y0), r
    @staticmethod
    def minCircleFor3Point(A, B, C):
        cosA = (B-A).dot(C-A)/(Point.dist(A, B)*Point.dist(A, C))
        cosB = (C-B).dot(A-B)/(Point.dist(B, A)*Point.dist(B, C))
        cosC = (A-C).dot(B-C)/(Point.dist(C, A)*Point.dist(C, B))
        if(cosA*cosB*cosC >= 0): #锐角/直角三角形采用外接圆
            center, radius = Point.circumcircle(A, B, C)
            return MinCircle(center, radius, [A, B, C], [True, True, True], True)
        else:# 钝角三角形用最长边做直径
            if(cosA < 0):
                return MinCircle(Point.mid(B, C), Point.dist(B, C)/2, [A, B, C], [False, True, True], False)
            elif(cosB < 0):
                return MinCircle(Point.mid(A, C), Point.dist(A, C)/2, [A, B, C], [True, False, True], False)
            else:
                return MinCircle(Point.mid(A, B), Point.dist(A, B)/2, [A, B, C], [True, True, False], False)
    @staticmethod
    def nextCircle(A, B, C, D):
        circles = []
        circles.append(Point.minCircleFor3Point(A, B, D))
        circles.append(Point.minCircleFor3Point(A, C, D))
        circles.append(Point.minCircleFor3Point(B, C, D))
        res = circles.copy()
        for circle in circles:
            p = circle.getPointNotInOriginPoints([A, B, C, D])
            #print('     new_circle:', circle)
            #print('     out_point:', p)
            if not circle.inCircle(p):
                res.remove(circle)
                #print('     oooooooooooooooout!')
        if len(res) > 0:
            return reduce(lambda a,b: a if a.radius < b.radius else b, res)
        return None
    @staticmethod
    def minCircle(points):
        points = list(set(points))
        #print(points)
        if len(points) is 0:
            return None, sys.maxsize
        elif len(points) is 1:
            return points[0], 0
        elif len(points) is 2:
            return Point.mid(points[0], points[1]), Point.dist(points[0], points[1])/2
        elif len(points) is 3:
            circle = Point.minCircleFor3Point(points[0], points[1], points[2])
            return circle.center, circle.radius
        ps = [points[0], points[1], points[2]]
        circle = Point.minCircleFor3Point(ps[0], ps[1], ps[2])
        while(True):
            maxInd = 0
            for i in range(1, len(points)):
                if Point.dist(circle.center, points[i]) > Point.dist(circle.center, points[maxInd]):
                    maxInd = i
            #print('circle:', circle)
            #print('point:', points[maxInd])
            #print('dist:', Point.dist(circle.center, points[maxInd]))
            if Point.dist(circle.center, points[maxInd]) <= circle.radius+1e-7:
                return circle.center, circle.radius
            circle = Point.nextCircle(ps[0], ps[1], ps[2], points[maxInd])
            if circle is None:
                return None, sys.maxsize
            ps = circle.points
    @staticmethod
    def toPoint(p):
        return Point(p[0], p[1])
    @staticmethod
    def toPointList(ps):
        return [Point.toPoint(p) for p in ps]
    def toNumpy(self):
        return np.array([self.x, self.y])


# In[10]:


def estimateKeypoint(points, d0):
    center, radius = Point.minCircle(Point.toPointList(points))
    if radius <= d0:
        return np.mean(points, axis=0)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    ind = np.argmax(np.bincount(kmeans.labels_))
    return kmeans.cluster_centers_[ind]


# In[11]:


if __name__ == '__main__':
    points = np.array([[26, 10],[29, 11],[26, 11],[29, 11], [30, 11], [29, 10], [29, 13], [30, 11]])
    print(estimateKeypoint(points, 3))


# In[12]:


def getAngle(points):
    line = points[0] - points[4]
    yaxis = np.array([0, -1])
    angle = np.rad2deg(np.arccos(np.dot(line, yaxis) / np.sqrt(np.sum(line**2))))
    if line[0] < 0:
        angle = -angle
    return angle


# In[13]:


def centralize(img, keypoints, length, origin_img_size, dest_img_size, channel):
    height = np.sqrt(np.sum((keypoints[0] - keypoints[4])**2))
    angle = getAngle(keypoints)
    img, keypoints = rotateImageAndPoints(img, keypoints, angle, origin_img_size, length / height)
    img = img.reshape(origin_img_size, origin_img_size, channel)
    dx = (dest_img_size - length + 1) // 2 - keypoints[0][1]
    dy = (dest_img_size + 1) // 2 - keypoints[0][0]
    for p in keypoints:
        p[1] = p[1] + dx
        p[0] = p[0] + dy
    transform = np.zeros((origin_img_size, origin_img_size, channel))
    for x in range(origin_img_size):
        for y in range(origin_img_size):
            for c in range(channel):
                if x-dx >= 0 and x-dx < origin_img_size and y-dy >= 0 and y-dy < origin_img_size:
                    transform[x][y][c] = img[x-dx][y-dy][c]
                else:
                    transform[x][y][c] = 0
    transform = transform[0:dest_img_size, 0:dest_img_size]
    return transform, keypoints


# In[14]:

"""
def getLabelBySeq(seq):
    m = ['Boeing', 'B-1', 'B-2', 'A-10', 'A-26', 'B-29', 'B-52', 'C-5', 'C-17', 'C-21', 
         'C-130', 'C-135', 'E-3', 'F-16', 'F-22', 'KC-10', 'P-63', 'T-6', 't-42', 'U-2']
    return m[seq]
"""

# In[ ]:

labels = ['Boeing', 'B-1', 'B-2', 'A-10', 'A-26', 'B-29', 'B-52', 'C-5', 'C-17', 'C-21', 
     'C-130', 'C-135', 'E-3', 'F-16', 'F-22', 'KC-10', 'P-63', 'T-6', 't-43', 'U-2']

def getLabelBySeq(seq):
    return labels[seq]

def getSeqByLabel(label):
    return labels.index(label)


# In[ ]:


def IOU(img, template):
    img = img.tolist()
    template = template.tolist()
    inter = 0
    union = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] != 0 and template[i][j] != 0:
                inter = inter + 1
            if img[i][j] != 0 or template[i][j] != 0:
                union = union + 1
    union = union + inter
    return inter / union


