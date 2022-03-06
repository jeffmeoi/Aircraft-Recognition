#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[2]:


resnet101_model = models.resnet101(pretrained=False)


# In[3]:


device = utils.device
# hyper parameters
learning_rate = 1e-4
batch_size = 16
epochs = 100
img_size = 224
n_labels = 20
model_file = 'resnet_recognize_aircraft_rotations.pth'


# In[4]:


class RecognitionNetwork(nn.Module):
    def __init__(self, num_classes=n_labels):
        super(RecognitionNetwork, self).__init__()
        self.resnet = resnet101_model
        self.fc1 = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        #print('x:', x.size())
        fc0 = self.resnet(x)
        #print('fc0:', fc0.size())
        fc1 = self.fc1(fc0)
        #print('fc1:', fc1.size())
        return fc1


# In[5]:


if __name__ == '__main__':
    # 随机生成输入数据
    rgb = torch.randn(4, 3, 224, 224)
    # 定义网络
    # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # channel,height,width用于第二个fc的reshape能匹配上pool5的输出shape
    # 默认输入图片数据大小为512*512
    net = RecognitionNetwork(num_classes=n_labels)
    # 模型参数过多，固化模型参数，降低内存损耗
    net.eval()
    # 前向传播
    out = net(rgb)
    # 打印输出大小
    print('-----'*5)
    print(out.shape)
    print('-----'*5)


# In[6]:

"""
def getImageAndPoints(annotation, img_dir, img_size=40):
    img_path = img_dir + annotation['label'] + '/' + annotation['name']
    img = cv2.imread(img_path)
    if(img is None):
        print(annotation, img_path)
    #print(img.shape)
    points = [float(x) for x in annotation['points'].split(',')]
    return img, points
"""
def getImageAndPoints(annotation, img_dir, img_size=40):
    img_path = img_dir + annotation['label'] + '/' + str(int(annotation['flip'])) + '.' + str(annotation['angle']) + '.' + annotation['name']
    img = cv2.imread(img_path)
    if(img is None):
        print(annotation, img_path)
    points = [float(x) for x in annotation['points'].split(',')]
    return img, points

# In[7]:


class TypeDataset(Dataset):
    def __init__(self, annotations_file=utils.annotations_dir + 'regular.json', 
                 img_dir=utils.rotation_dir_224):
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array([img, img, img])
        img = torch.from_numpy(img).float()#.permute(2,0,1)
        seq = self.annotations[idx]['seq']
        return img, seq


# In[8]:


train_data = TypeDataset(annotations_file=utils.annotations_dir + 'keypoints_train_set_224.json')
test_data = TypeDataset(annotations_file=utils.annotations_dir + 'keypoints_test_set_224.json')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[9]:


model = RecognitionNetwork(n_labels).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

def train_loop(dataloader, model, loss_fn, optimizer):
    model = model.train()
    size = len(dataloader.dataset)
    # x = (batch_num, channels, height, width)
    # y = (batch_num, height, width), type Long, value = [0, C), C is the num of classes.
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device)
        pred = model(x) # get the predict result pred = (batch_num, num_classes, height, width)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x) # get the predict result pred = (batch_num, num_classes, height, width)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Status: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[10]:


if(os.path.exists(model_file)):
    model.load_state_dict(torch.load(model_file, map_location='cpu'))


# In[11]:


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


# In[12]:


if __name__ == '__main__':
    train()
   # pass

# In[26]:


def evals():
    model.eval()
    cnt = 0
    for batch, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = pred.argmax(1)
        for i in range(batch_size):
            plt.imshow(x[i,:].permute(1,2,0).cpu().numpy().astype(np.uint8))
            plt.show()
            print((pred[i] == y[i]).item())
            if pred[i] == y[i]:
                cnt = cnt + 1
    print(cnt / len(test_data))



# In[27]:


if __name__ == '__main__':
    evals()

# In[ ]:



def predict(img):
    model.eval()
    x = torch.unsqueeze(torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1), 0)
    x = x.to(device)
    pred = model(x).argmax(1)
    return pred.item()
