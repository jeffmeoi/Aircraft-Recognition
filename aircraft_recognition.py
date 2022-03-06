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
import time
import keypoints_detect_network_224 as keypoints_detect_network


# In[2]:

vgg16_model = models.vgg16(pretrained=False)


# In[3]:


def decoder(input_channel, output_channel, num=3):
    if num == 3:
        decoder_body = nn.Sequential(
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1))
    elif num == 2:
        decoder_body = nn.Sequential(
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1))
    return decoder_body


# In[4]:


class CoarseSegmentNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(CoarseSegmentNetwork, self).__init__()
        # input 224*224*3
        self.encoder1 = vgg16_model.features[:4]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True) #112*112*64
        
        self.encoder2 = vgg16_model.features[5:9]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True) #56*56*128
        
        self.encoder3 = vgg16_model.features[10:16]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True) #28*28*256

        self.encoder4 = vgg16_model.features[17:23]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True) #14*14*512

        self.encoder5 = vgg16_model.features[24:30]
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True) #7*7*512
        
        self.classifier = nn.Sequential(
            torch.nn.Linear(int(224*224/2), 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, int(224*224/2)),
            torch.nn.ReLU(),
        )

        self.decoder5 = decoder(512, 512)
        self.unpool5 = nn.MaxUnpool2d(2, 2)

        self.decoder4 = decoder(512, 256)
        self.unpool4 = nn.MaxUnpool2d(2, 2)

        self.decoder3 = decoder(256, 128)
        self.unpool3 = nn.MaxUnpool2d(2, 2)

        self.decoder2 = decoder(128, 64, 2)
        self.unpool2 = nn.MaxUnpool2d(2, 2)

        self.decoder1 = decoder(64, num_classes, 2)
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        
    def forward(self, x):
        #print('x:', x.size())
        encoder1 = self.encoder1(x);
        #print('encoder1:', encoder1.size())
        output_size1 = encoder1.size()
        pool1, indices1 = self.pool1(encoder1)
        #print('pool1:', pool1.size());
        #print('indices1:', indices1.size())

        encoder2 = self.encoder2(pool1);
        #print('encoder2:', encoder2.size())
        output_size2 = encoder2.size()
        pool2, indices2 = self.pool2(encoder2)
        #print('pool2:', pool2.size())
        #print('indices2:', indices2.size())

        encoder3 = self.encoder3(pool2)
        #print('encoder3:', encoder3.size())

        output_size3 = encoder3.size()
        pool3, indices3 = self.pool3(encoder3)
        #print('pool3:', pool3.size())
        #print('indices3:', indices3.size())

        encoder4 = self.encoder4(pool3)
        #print('encoder4:', encoder4.size())
        
        output_size4 = encoder4.size()
        pool4, indices4 = self.pool4(encoder4)
        #print('pool4:', pool4.size())
        #print('indices4:', indices4.size())

        encoder5 = self.encoder5(pool4)
        #print('encoder5:', encoder5.size())
        output_size5 = encoder5.size()
        
        pool5, indices5 = self.pool5(encoder5)
        #print('pool5:', pool5.size())
        #print('indices5:', indices5.size())

        pool5 = pool5.view(pool5.size(0), -1)
        #print('pool5:', pool5.size())
        fc = self.classifier(pool5)
        #print('fc:', fc.size())
        fc = fc.reshape(pool5.size(0), 512, 7, 7)
        #print('fc:', fc.size()) 

        unpool5 = self.unpool5(input=fc, indices=indices5, output_size=output_size5)
        #print('unpool5:', unpool5.size())
        decoder5 = self.decoder5(unpool5)
        #print('decoder5:', decoder5.size())

        unpool4 = self.unpool4(input=decoder5, indices=indices4, output_size=output_size4)
        #print('unpool4:', unpool4.size())
        decoder4 = self.decoder4(unpool4)
        #print('decoder4:', decoder4.size())

        unpool3 = self.unpool3(input=decoder4, indices=indices3, output_size=output_size3)
        #print('unpool3:', unpool3.size())
        decoder3 = self.decoder3(unpool3)
        #print('decoder3:', decoder3.size())

        unpool2 = self.unpool2(input=decoder3, indices=indices2, output_size=output_size2)
        #print('unpool2:', unpool2.size())
        decoder2 = self.decoder2(unpool2)
        #print('decoder2:', decoder2.size())

        unpool1 = self.unpool1(input=decoder2, indices=indices1, output_size=output_size1)
        #print('unpool1:', unpool1.size())
        decoder1 = self.decoder1(unpool1)
        #print('decoder1:', decoder1.size())

        return decoder1


# In[5]:


if __name__ == '__main__':
    #print(vgg16_model.features)
    # 随机生成输入数据
    rgb = torch.randn(4, 3, 224, 224)
    # 定义网络
    # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # channel,height,width用于第二个fc的reshape能匹配上pool5的输出shape
    # 默认输入图片数据大小为512*512
    net = CoarseSegmentNetwork(num_classes=2)
    # 模型参数过多，固化模型参数，降低内存损耗
    net.eval()
    # 前向传播
    out = net(rgb)
    # 打印输出大小
    print('-----'*5)
    print(out.shape)
    print('-----'*5)


# In[6]:


device = utils.device
# hyper parameters
learning_rate = 1e-4
batch_size = 4
epochs = 30
img_size=224
weights=torch.tensor([0.92, 0.08]).to(device)
n_labels = 2
model_file = '19_coarse_segment_network.pth'
batch_count = 16

# In[ ]:


model = CoarseSegmentNetwork().to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    model = model.train()
    tot_loss = 0
    size = len(dataloader.dataset)
    # x = (batch_num, channels, height, width)
    # y = (batch_num, height, width), type Long, value = [0, C), C is the num of classes.
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device)
        pred = model(x) # get the predict result pred = (batch_num, num_classes, height, width)
        loss = loss_fn(pred, y)
        tot_loss = tot_loss + loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_count  == 0:
            avg_loss, current = tot_loss / batch_count, batch * len(x)
            print(f"avg loss: {avg_loss:>7f}  [{current:>5d}/{size:>5d}]")
            tot_loss = 0

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
    correct /= size*img_size*img_size
    print(f"Test Status: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[ ]:

def getImageAndLabel(annotation, img_dir):
    img_path = img_dir + annotation['label'] + '/' + annotation['name']
    img = cv2.imread(img_path)
    if(img is None):
        print(annotation, img_path)
    img_shape = img.shape
    img = utils.resize(img, img_size, img_size)

    #锐化
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    #img = cv2.filter2D(img, -1, kernel=kernel)
    
    # 对比度调整
    # img = cv2.normalize(img,dst=None,alpha=280,beta=10,norm_type=cv2.NORM_MINMAX)
    
    img = torch.from_numpy(img).float()
    img = img.permute(2,0,1)
    label = np.zeros(img_shape[:2])
    points = np.array(annotation['points'])
    cv2.fillPoly(label, [points], 1)
    label = utils.resize(label, img_size, img_size)
    label =  torch.from_numpy(label).long()
    return img, label


# In[ ]:


class BitmapDataset(Dataset):
    def __init__(self, annotations_file=utils.annotations_dir + 'annotations.xml', isXml=True, 
                 img_dir=utils.img_dir):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.annotations = list(utils.getAnnotations(url=annotations_file, isXml=isXml))
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return getImageAndLabel(self.annotations[idx], self.img_dir)


# In[ ]:

train_data = BitmapDataset(annotations_file=utils.annotations_dir + 'train_set.json', isXml=False)
test_data = BitmapDataset(annotations_file=utils.annotations_dir + 'test_set.json', isXml=False)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# In[ ]:


if(os.path.exists(model_file)):
    #model.load_state_dict(torch.load(model_file, map_location='cpu'))
    pass

# In[ ]:


def train():
    print('train start:', time.time())
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        if epoch % 5 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    torch.save(model.state_dict(), model_file)


# In[ ]:


def evals():
    model.eval()
    for batch, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = nn.functional.log_softmax(pred, dim=1)
        pred = pred.argmax(1)
        for i in range(4):
            plt.imshow(x[i, 0, :].cpu().numpy())
            plt.show()
            plt.imshow(pred[i,:].cpu().numpy())
            plt.show()
        if(batch >= 5):
            break


# In[ ]:

def saturability(img, sat=0.50):
    img = img.astype(np.float32)

    r = img[:,:,2]
    g = img[:,:,0]
    b = img[:,:,1]

    m, n, c = img.shape
    r_new = r
    g_new = g
    b_new = b
    
    img_new = np.zeros(img.shape)

    increment = sat

    for i in range(m):
        for j in range(n):
            rgbmax = max(r[i,j], max(g[i, j], b[i, j]))
            rgbmin = min(r[i,j], min(g[i, j], b[i, j]))
            delta = (rgbmax-rgbmin)/255
            if delta==0: continue
            value = (rgbmax+rgbmin)/255
            L = value/2

            if L<0.5: S = delta/value
            else: S=delta/(2-value)

            if increment>=0:
                if (increment+S)>=1:
                    alpha = S
                else:
                    alpha = 1-increment
                alpha =1/alpha-1
                r_new[i, j] = r[i, j] +(r[i,j]-L*255)*alpha
                g_new[i, j] = g[i, j] +(g[i,j]-L*255)*alpha
                b_new[i, j] = b[i, j] +(b[i,j]-L*255)*alpha
            else:
                alpha = increment
                r_new[i, j] = L*255 +(r[i,j]-L*255)*alpha
                g_new[i, j] = L*255 +(g[i,j]-L*255)*alpha
                b_new[i, j] = L*255 +(b[i,j]-L*255)*alpha

    img_new[:,:,2]=r_new
    img_new[:,:,1]=b_new
    img_new[:,:,0]=g_new        

    return img_new

def coarseSegment(img):
    x = torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), 0).to(device)
    pred = model(x)
    pred = nn.functional.softmax(pred, dim=1)
    return pred.cpu().detach().numpy()
def crf(img, pred, time=10):
    # get unary potentials (neg log probability)
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
    #U = unary_from_labels(pred, n_labels, gt_prob=0.7, zero_unsure=False)
    U = unary_from_softmax(pred.reshape([2, -1]), scale=0.6)
    d.setUnaryEnergy(U)
    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(2, 2), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(12,12,12),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(time)
    # Find out the most probable class for each pixel.
    return Q


# In[ ]:

def predict(img, pred=None):
    if not isinstance(pred, np.ndarray):
        pred = coarseSegment(img)
        
    #锐化
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    #img = cv2.filter2D(img, -1, kernel=kernel)
    
    #plt.imshow(img.astype(np.uint8))
    #plt.show()
    
    
    # 对比度调整
    img = cv2.normalize(img,dst=None,alpha=280, beta=80,norm_type=cv2.NORM_MINMAX)
    
    #img_cp = img.copy()
    #plt.imshow(img.astype(np.uint8))
    #plt.show()
    
    # 饱和度调整
    img = saturability(img, 20)

    #plt.imshow(img.astype(np.uint8))
    #plt.show()
    
    Q = crf(img, pred, 10)
    #plt.imshow(np.argmax(Q, axis=0).reshape(224, 224, -1))
    #plt.show()
    #Q = np.array(Q).reshape(2, 224, 224)
    #Q = crf(img_cp, Q, 1)
    #plt.imshow(np.argmax(Q, axis=0).reshape(224, 224, -1))
    #plt.show()
    #print(dataset.annotations[i]['label'], dataset.annotations[i]['name'])

    #plt.imshow(img.astype(np.int32))
    #plt.show()
    #pred = torch.argmax(pred, dim=1)
    #plt.imshow(np.squeeze(pred.detach().numpy()))
    #plt.show()
    #plt.imshow(MAP.reshape(224, 224, -1))
    #plt.show()
    
    MAP = np.argmax(Q, axis=0)
    MAP =  MAP.reshape(224, 224, -1)
    return MAP


    
def total_crf():
    for i in range(len(train_data)):
        x, y = train_data[i]
        img = x.permute(1, 2, 0).cpu().detach().numpy()
        #print(train_data.annotations[i]['label'], train_data.annotations[i]['name'])
        #plt.imshow(img.astype(np.uint8))
        #plt.show()
        segment = predict(img)
        segment = (segment*255).astype(np.uint8)
        angle, keypoints = keypoints_detect_network.getAngleAndKeypoints(utils.resize(img, 224, 224))
        #height = np.sqrt(np.sum((keypoints[0] - keypoints[4])**2))
        
        transform, keypoints = utils.centralize(segment, keypoints, 75, 224, 224, 1)
        #plt.imshow(transform)
        #plt.show()
        if not os.path.exists(utils.crf_dir + train_data.annotations[i]['label']):
            os.mkdir(utils.crf_dir + train_data.annotations[i]['label'])
        cv2.imwrite(utils.crf_dir + train_data.annotations[i]['label'] + '/' + train_data.annotations[i]['name'], transform)
    for i in range(len(test_data)):
        x, y = test_data[i]
        img = x.permute(1, 2, 0).cpu().detach().numpy()
        #print(test_data.annotations[i]['label'], test_data.annotations[i]['name'])
        #plt.imshow(img)
        #plt.show()
        segment = predict(img)
        segment = (segment*255).astype(np.uint8)
        angle, keypoints = keypoints_detect_network.getAngleAndKeypoints(utils.resize(img, 224, 224))
        # height = np.sqrt(np.sum((keypoints[0] - keypoints[4])**2))
        transform, keypoints = utils.centralize(segment, keypoints, 75, 224, 224, 1)
        if not os.path.exists(utils.crf_dir + test_data.annotations[i]['label']):
            os.mkdir(utils.crf_dir + test_data.annotations[i]['label'])
        #plt.imshow(transform)
        #plt.show()
        cv2.imwrite(utils.crf_dir + test_data.annotations[i]['label'] + '/' + test_data.annotations[i]['name'], transform)



# In[ ]:


if __name__ == '__main__':
    #train()
    total_crf()


# In[ ]:




