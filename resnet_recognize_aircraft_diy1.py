#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, io
import numpy as np
import cv2
import utils
import json
from matplotlib import pyplot as plt
from custom_model_type_recognition1 import CustomModel, BasicBlock, Bottleneck


# In[2]:


device = utils.device
# hyper parameters
learning_rate = 1e-4
batch_size = 64
epochs = 200
img_size=224
n_labels = 20
model_file = 'resnet_recognize_aircraft_diy1.pth'


# In[3]:


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
class RecognitionNetwork(nn.Module):
    def __init__(self, num_classes=20):
        super(RecognitionNetwork, self).__init__()
        self.conv1 = conv(1, 32)
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
        self.fc1 = nn.Linear(128, num_classes)
        
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
class RecognitionNetwork(nn.Module):
    def __init__(self, num_classes=n_labels):
        super(RecognitionNetwork, self).__init__()
        self.model = CustomModel(BasicBlock, [2,2,2,2], num_classes=num_classes, channels=1)

    def forward(self, x):
        return self.model(x)

# In[5]:


if __name__ == '__main__':
    # 随机生成输入数据
    rgb = torch.randn(4, 1, 224, 224)
    # 定义网络
    # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # channel,height,width用于第二个fc的reshape能匹配上pool5的输出shape
    # 默认输入图片数据大小为224*224
    net = RecognitionNetwork(num_classes=10)
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
    img_path = img_dir + annotation['label'] + '/' + annotation['name']
    img = np.expand_dims(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), axis=2)
    if(img is None):
        print(annotation, img_path)
    points = [float(x) for x in annotation['points'].split(',')]
    return img, points


# In[7]:


class TypeDataset(Dataset):
    def __init__(self, annotations_file=utils.annotations_dir + 'regular.json', 
                 img_dir=utils.regular_dir):
        regular_file = open(annotations_file, 'r')
        regular_str = regular_file.read()
        annotations = json.loads(regular_str)
        regular_file.close()
        
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.annotations = annotations
        
        self.dict = []
        for anno in annotations:
            img, points = getImageAndPoints(anno, self.img_dir, img_size)
            seq = anno['seq']
            self.dict.append((torch.from_numpy(img).float().permute(2,0,1), seq))
            flip_img = cv2.flip(img, 1)
            self.dict.append((torch.tensor(np.array([flip_img])).float(), seq))

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        return self.dict[idx]


# In[8]:


train_data = TypeDataset(annotations_file=utils.annotations_dir + 'regular_train_set.json')
test_data = TypeDataset(annotations_file=utils.annotations_dir + 'regular_test_set.json')
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

        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size
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
        if epoch % 10 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        train_loop(train_dataloader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), model_file)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


# In[ ]:


if __name__ == '__main__':
    train()


# In[ ]:


def evals():
    model.eval()
    for batch, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = pred.argmax(1)
        for i in range(batch_size):
            plt.imshow(x[i,:].permute(1,2,0).cpu().numpy().astype(np.uint8))
            plt.show()
            print((pred[i] == y[i]).item())
        if(batch >= 5):
            break


# In[ ]:


if __name__ == '__main__':
    evals()


# In[ ]:


def predict(img):
    model.eval()
    x = torch.unsqueeze(torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1), 0).to(device)
    pred = model(x)
    seq = pred.argmax(1).item()
    return seq, pred[0][seq]

