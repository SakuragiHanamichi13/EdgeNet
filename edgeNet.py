import numpy as np
import cv2
import os
import pandas as pd
#pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F



def data_info(folder, xName = 'x/', yName='y/'):
    output = {'imgPath':[], 'target':[]}
    for root, dir, img_names in os.walk(folder):
        for img_name in img_names:
            if img_name[-1]=='g':
                img_path = folder + xName + img_name
                edge_path = folder + yName + img_name
                output['imgPath'].append(img_path)
                output['target'].append(edge_path)
    output = pd.DataFrame(output)

    return output


class data(Dataset):
    
    def __init__(self, img_folder, xName = 'x/', yName='y/', transforms=None):
        self.transforms = transforms
        self.data_info = data_info(img_folder)
        print(type(data_info))
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        x_name = self.data_info['imgPath'][idx]
        y_name = self.data_info['target'][idx]
        img = cv2.imread(x_name, 0)
        tar = cv2.imread(y_name, 0)
        sample = {'img':img, 'tar':tar}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def summary(self):
        display(self.data_info.head())
        
        print('transform = ', self.transforms)


class ToTensor(object):
    '''convert ndarrays in sample to Tensors'''
    def __call__(self, sample):
        img, tar = sample['img']/255.0, sample['tar']/255.0
        img = np.expand_dims(img, axis=2)
        #tar = np.expand_dims(tar, axis=2)
        #swap color axis because
        #numpy image: H x W x C
        #torch image: Cx H x W
        img = img.transpose((2,0,1))
        #tar = tar.transpose((2,0,1))
        return {'img':torch.from_numpy(img), 'tar':torch.from_numpy(tar)}

def conv3(in_channels, out_channels, stride=1, padding=1, bias=True):
    '''3x3 convolution with padding and bias'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride,padding=padding, bias=bias)
def conv1(in_channels, out_channels, stride=1, bias=True):
    '''1x1 convolution with bias'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride,bias=bias)

def bn(channels):
    return nn.BatchNorm2d(channels)



def block(in_3, out_3, out_1, batch_norm=False):
    layers = []
    if batch_norm:
        layers+=[conv3(in_3, out_3), conv1(out_3, out_1),bn(out_1),  nn.ReLU(inplace=True)]
    else:
        layers+=[conv3(in_3, out_3), conv1(out_3, out_1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def make_blocks(cfg, batch_norm=False):
    blocks = []
    
    for v in cfg:
        bk = block(v[0], v[1], v[2], batch_norm)
        blocks.append(bk)
    return nn.Sequential(*blocks) 


class net(nn.Module):
    
    def __init__(self, bks):
        super(net, self).__init__()
        self.bks = bks
        self.reduction = conv1(bks[-1][-3].out_channels, 2)
        
    def forward(self, x):
        x = self.bks(x)
        x = self.reduction(x)
        return F.log_softmax(x, dim=1)
    


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    num_data = len(train_loader.dataset)
    for batch_i, samples_batched in enumerate(train_loader):
        '''
        samples_batched['img'], samples_batched['tar']
        '''
        x = samples_batched['img'].float()
        y = samples_batched['tar'].long()
        size_batch = x.size()[0]
        #data to device
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        #print('pred size = ',  output.size())
        #print('tar size = ', y.size())
        loss = criterion(output, y)#average batch loss
        loss.backward()
        optimizer.step()

        #loss
        train_loss+=size_batch*loss.item()#add sum of loss in the current batch

    average_loss_epoch = train_loss/num_data
    return average_loss_epoch

def val(model, device, test_loader, criterion, earily_stopping = 0.08):
    model.eval()
    val_loss = 0
    num_data = len(test_loader.dataset)
    
    keep_training = False
       
    with torch.no_grad():
        for batch_i, samples_batched in enumerate(test_loader):
            x = samples_batched['img'].float()
            y = samples_batched['tar'].long()
            size_batch = x.size()[0]

            x, y = x.to(device), y.to(device)
            output = model(x)
            
            loss = criterion(output, y)
            val_loss+=size_batch*loss.item()

        average_loss_epoch = val_loss/num_data
    if average_loss_epoch>earily_stopping:
        keep_training = True
    return average_loss_epoch, keep_training


kers = {'sobel_x':np.array([[-1,0,1], [-2,0,2], [-1,0,1]]),
        'sobel_y':np.array([[1,2,1], [0,0,0], [-1,-2,-1]]),
        'median':np.array([[1.0/3,1.0/3,1.0/3], [1.0/3,1.0/3,1.0/3], [1.0/3,1.0/3,1.0/3]])}


def summary_conv(conv):
    assert isinstance(conv, nn.Conv2d), 'input must be of the type nn.Conv2d'
    
    summary = {}
    summary.update({'in_channels':conv.in_channels})
    summary.update({'out_channels': conv.out_channels})
    summary.update({'weight size':conv.weight.size()}) 
    summary.update({'bias size':conv.bias.size()})
    
    display(summary)



def init_bias(conv, value):
    for i in range(conv.bias.size()[0]):
        conv.bias[i] = torch.tensor(value).clone()
        


def init_weights(conv, ker_np):
    assert isinstance(conv, nn.Conv2d), 'input must be of the type nn.Conv2d'
    
    '''
    conv.weight is of size [out_channels, in_channels, kernel_size[0], kernel_size[1]]
    '''
    
    ker_ten = torch.tensor(ker_np)
    for out_i in range(conv.out_channels):
        for in_j in range(conv.in_channels):
            conv.weight[out_i, in_j, :, :] = ker_ten.clone()



 

def init_conv3x3(conv, ker_np, value):
    assert isinstance(conv, nn.Conv2d), 'input must be of the type nn.Conv2d'

    init_bias(conv, value)
    init_weights(conv, ker_np)

#def init_block(block, ):






















