#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, io
from annotations import getLabels as getAnnotations
import numpy as np
from matplotlib import pyplot as plt
import cv2
import json
import utils
from custom_model_keypoint_detect import CustomModel, BasicBlock, Bottleneck

# In[2]:


device = utils.device
learning_rate = 1e-4
batch_size = 128
epochs = 100
img_size = 224
model_file = 'keypoints_detect_network_diy.pth'


# In[3]:

"""
def conv(in_channels, out_channels, stride=1):
    res = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(stride,stride), padding=(1,1)),
        nn.ReLU(inplace=True),
    )
    return res
def downsample(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(2,2), padding=(0,0))
"""
def conv(in_channels, out_channels, stride=1, has_relu=True):
    res = None
    if has_relu:
        res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(stride,stride), padding=(1,1)),
            nn.ReLU(inplace=True),
        )
    else:
        res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(stride,stride), padding=(1,1)),
        )
    return res

def downsample(in_channels, out_channels, s=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(s,s), padding=(0,0))


# In[4]:

"""
class KeypointsDetectNetwork(nn.Module):
    def __init__(self):
        super(KeypointsDetectNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.fc1 = nn.Linear(1000, 16)

    def forward(self, x):
        fc0 = self.resnet(x)
        fc1 = self.fc1(fc0)
        return fc1
"""
"""
class KeypointsDetectNetwork(nn.Module):
    def __init__(self):
        super(KeypointsDetectNetwork, self).__init__()
        self.conv1 = conv(3, 32)
        self.conv2 = conv(32, 32)
        self.conv3 = conv(32, 32, has_relu=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv(32, 64)
        self.conv5 = conv(64, 64, has_relu=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = conv(64, 128)
        self.conv7 = conv(128, 128, has_relu=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc0 = nn.Linear(100352, 128)
        self.fc1 = nn.Linear(128, 16)

        self.downsample1 = downsample(32, 64, 1)
        self.downsample2 = downsample(64, 128, 1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('x:', x.size())
        conv1 = self.conv1(x)
        #print('conv1', conv1.size())
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2) + conv1
        conv3 = self.relu1(conv3)
        pool1 = self.pool1(conv3)
        #print('pool1', pool1.size())

        downsample1 = self.downsample1(pool1)
        conv4 = self.conv4(pool1)
        conv5 = self.conv5(conv4) + downsample1
        conv5 = self.relu2(conv5)
        pool2 = self.pool2(conv5)
        #print('pool2', pool2.size())

        downsample2 = self.downsample2(pool2)
        conv6 = self.conv6(pool2)
        conv7 = self.conv7(conv6) + downsample2
        conv7 = self.relu3(conv7)
        pool3 = self.pool3(conv7)
        #print('pool3', pool3.size())

        flat = pool3.contiguous().view(x.size(0), -1)
        fc0 = self.fc0(flat)
        #print('fc0:', fc0.size())
        fc1 = self.fc1(fc0)
        #print('fc1', fc1.size())
        return fc1
"""
class KeypointsDetectNetwork(nn.Module):
    def __init__(self):
        super(KeypointsDetectNetwork, self).__init__()
        self.model = CustomModel(BasicBlock, [2,2,2,2], num_classes=16, channels=3)

    def forward(self, x):
        return self.model(x)

# In[5]:


if __name__ == '__main__':
    # 随机生成输入数据
    rgb = torch.randn(1, 3, img_size, img_size)
    # 定义网络
    # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # channel,height,width用于第二个fc的reshape能匹配上pool5的输出shape
    # 默认输入图片数据大小为512*512
    net = KeypointsDetectNetwork()
    # 模型参数过多，固化模型参数，降低内存损耗
    net.eval()
    # 前向传播
    out = net(rgb)
    # 打印输出大小
    print('-----'*5)
    print(out.shape)
    print('-----'*5)


# In[6]:


def getImageAndPoints(annotation, img_dir, img_size=40):
    img_path = img_dir + annotation['label'] + '/' + str(int(annotation['flip'])) + '.' + str(annotation['angle']) + '.' + annotation['name']
    img = cv2.imread(img_path)
    if(img is None):
        print(annotation, img_path)
    points = [float(x) for x in annotation['points'].split(',')]
    return img, points


# In[7]:


class KeypointDataset(Dataset):
    def __init__(self, annotations_file=utils.annotations_dir + 'rotations.json', img_dir=utils.rotation_dir_224):
        rotations_file = open(annotations_file, 'r')
        rotations_str = rotations_file.read()
        annotations = json.loads(rotations_str)
        rotations_file.close()
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img, points = getImageAndPoints(self.annotations[idx], self.img_dir, img_size=img_size)
        img = torch.from_numpy(img).float().permute(2,0,1)
        points = torch.tensor(points)
        return img, points


# In[8]:


train_data = KeypointDataset(annotations_file=utils.annotations_dir + 'keypoints_train_set_224.json')
test_data = KeypointDataset(annotations_file=utils.annotations_dir + 'keypoints_test_set_224.json')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[9]:


if __name__ == '__main__':
    plt.imshow(train_data[0][0].permute(1,2,0).int())
    plt.show()
    print(train_data[0][1])


# In[10]:


model = KeypointsDetectNetwork().to(device)
def loss_fn(pred, y):
    top = y[:, 0:2]
    #left = y[:, 4:6]
    bottom = y[:, 8:10]
    #right = y[:, 12:14]
    height = nn.functional.pairwise_distance(top, bottom)
    #width = nn.functional.pairwise_distance(left, right)
    distances = nn.functional.pairwise_distance(pred, y)
    return torch.mean(torch.pow(distances / height, 2)) / 8
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batch_count = 20
def train_loop(dataloader, model, loss_fn, optimizer):
    print('Start Train Loop')
    model = model.train()
    size = len(dataloader.dataset)
    # x = (batch_num, channels, height, width)
    # y = (batch_num, height, width), type Long, value = [0, C), C is the num of classes.
    train_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device)
        pred = model(x) # get the predict result pred = (batch_num, num_classes, height, width)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
        if batch % batch_count == 0 and batch is not 0:
            current = batch * len(x)
            print(f"Avg loss: {(train_loss / batch_count):>7f}  [{current:>5d}/{size:>5d}]")
            train_loss = 0


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    print('Start Test Loop')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x) # get the predict result, pred = (batch_num, num_classes, height, width)
            test_loss += loss_fn(pred, y).item()
    test_loss = test_loss * batch_size / size
    print(f"Test Status: \n Avg loss: {test_loss:>8f} \n")


# In[11]:


if(os.path.exists(model_file)):
    model.load_state_dict(torch.load(model_file, map_location='cpu'))


# In[12]:


def train():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        if epoch % 5 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        train_loop(train_dataloader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), model_file)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


# In[ ]:


if __name__ == '__main__':
    train()
    #pass


# In[ ]:


def rotate(img):        # img is (3, 40, 40)
    l = []
    for i in range(4):
        l.append(utils.rotateImageAndPoints(img, None, 90*i, img_size, 1)[0])
    img=cv2.flip(img,1)
    for i in range(4):
        l.append(utils.rotateImageAndPoints(img, None, 90*i, img_size, 1)[0])
    return torch.tensor(np.stack(l, axis=0)).permute(0,3,1,2)
# imgs=(8,40,40,3), preds=(8,8,2)
def rotateBack(pred):
    res = []
    for i in range(8):
        res.append(utils.rotateImageAndPoints(None, pred[i], -90*(i%4), img_size, 1)[1])
    for i in range(4, 8):
        for point in res[i]:
            point[0] = img_size - point[0]
    return np.stack(res, axis=0)
def getAngleAndKeypoints(img):
    model.eval()
    imgs = rotate(img).to(device)
    img = utils.resize(img, img_size, img_size)
    pred = model(imgs)
    pred = pred.reshape(pred.size(0), -1, 2).cpu().detach().numpy().astype(np.int32)
    """
    for j in range(8):
        img = imgs[j,:].cpu().permute(1,2,0).numpy().astype(np.uint8).copy()
        cv2.polylines(img, [pred[j, :]], True, (255, 0, 0))
        plt.imshow(img)
        plt.show()
    """
    pred = rotateBack(pred)
    res = []
    for j in range(8):
        res.append(utils.estimateKeypoint(pred[:,j,:], 3))
    res = np.stack(res,axis=0).astype(np.int32)
    im = imgs[0,:].cpu().permute(1,2,0).numpy().astype(np.uint8).copy()
    angle = utils.getAngle(res)
    """
    im, res = utils.rotateImageAndPoints(im, res, angle, img_size)
    cv2.polylines(im, [res], True, (255, 0, 0))
    plt.imshow(im)
    plt.show()
    """
    return angle, res


# In[ ]:


def train_eval():
    model.eval()
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = pred.reshape(pred.size(0), -1, 2).cpu().detach().numpy().astype(np.int32)
        for i in range(x.size(0)):
            img = x[i,:].permute(1,2,0).cpu().numpy().astype(np.int32).copy()
            cv2.polylines(img, [pred[i]], True, (255, 0, 0))
            plt.imshow(img)
            plt.show()
        break
def test_eval():
    model.eval()
    for batch, (x, y) in enumerate(test_dataloader):
        for i in range(x.size(0)):
            angle, points = getAngleAndKeypoints(x[i, :].cpu().permute(1,2,0).detach().numpy())
            print(angle)


# In[ ]:


if __name__ == '__main__':
    pass
    #test_eval()


# In[ ]:




