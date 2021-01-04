#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def my_cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
            pred: predictions for neural network
            targets: targets, can be soft
            size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    if True:
        return torch.mean(torch.sum(-target * logsoftmax(input) , dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        if self.args.loss == 'Custom':
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if idxs == None:
            self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
    

    def train(self, net):
        net.train()
        # train and update
        if self.args.opt == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=5e-4)

        #print('train starts here!')
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                #images, labels = images.to(self.args.device), labels.to(self.args.device)

                if self.args.gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                    # wrap them in Variable
                    images, labels = Variable(images), Variable(labels)

                net.zero_grad()
                log_probs = net(images)
                if self.args.loss == 'Custom':
                    onehot_labels = torch.nn.functional.one_hot(labels,num_classes=self.args.num_classes)
                    loss = my_cross_entropy(log_probs, onehot_labels)
                    #loss = self.loss_func(log_probs, onehot_labels)
                else:
                    loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                '''
                if self.args.verbose and batch_idx % 10 == 0: #self.args.verbose
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                '''
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_with_BACC(object):
    def __init__(self, args, dataset=None, label=None):
        self.args = args
        self.selected_clients = []
        self.X_train = dataset
        self.y_train = label
        self.sample_num = self.X_train.shape[0]

    def train(self, net):
        net.train()
        # train and update
        if self.args.opt == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=5e-4)

        #print('train starts here!')
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []

            batch_iter = int(self.sample_num/self.args.local_bs)

            for batch_idx in range(batch_iter):
                #images, labels = images.to(self.args.device), labels.to(self.args.device)
                stt_pos = batch_idx * self.args.local_bs
                end_pos = (batch_idx + 1) * self.args.local_bs
                images_np = self.X_train[stt_pos:end_pos,:]
                images_np = np.reshape(images_np, (self.args.local_bs,1,28,28))
                images = torch.Tensor(images_np)

                labels_np = self.y_train[stt_pos:end_pos,:]
                labels = torch.Tensor(labels_np)

                if self.args.gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                    # wrap them in Variable
                    images, labels = Variable(images), Variable(labels)

                net.zero_grad()
                log_probs = net(images)
                if self.args.loss == 'Custom':
                    #onehot_labels = torch.nn.functional.one_hot(labels,num_classes=self.args.num_classes)
                    loss = my_cross_entropy(log_probs, labels)
                    #loss = self.loss_func(log_probs, onehot_labels)
                else:
                    loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                '''
                if self.args.verbose and batch_idx % 10 == 0: #self.args.verbose
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                '''
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

