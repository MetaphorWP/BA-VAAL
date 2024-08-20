from prettytable import PrettyTable

import config
import torch
from typing import List
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

##
# Loss Prediction Loss

class ConfusionMatrix(object):
    def __init__(self, nums_classes: int, labels: List):
        self.matrix = np.zeros((nums_classes, nums_classes))
        self.nums_classes = nums_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self,alpha):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.nums_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print('the model accuracy is ', acc)


        table = PrettyTable()
        table.field_names = ['', 'Precision', 'Recall',]
        precision_list = []
        recall_list = []

        for i in range(self.nums_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP  # 真实类别为其他类，被识别为i类
            FN = np.sum(self.matrix[:, i]) - TP  # 真实类别为i类，被识别为其他类

            precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
            recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
            table.add_row([self.labels[i], precision, recall])
            precision_list.append(precision)
            recall_list.append(recall)

        std_re = np.std(np.array(recall_list))
        print(self.matrix)
        print(table)
        # pl = np.array(precision_list)
        rl = np.array(recall_list)
        # pl = pl.mean() - pl
        rl = rl.mean() - rl
        # old Ratio
        # ratio = 2 * (pl * rl) * alpha / (pl + rl)
        # new Ratio
        ratio = 2 * rl * alpha
        print("Ratio:{}".format(ratio))
        # ratio = np.where(ratio > 0,ratio,0.0)
        return ratio,std_re

def kl_div(source, target, reduction='batchmean'):
    loss = F.kl_div(F.log_softmax(source, 1), target, reduction=reduction)
    return loss

def LossPredLoss(input, target, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def test(models, dataloaders, alpha, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    cm = ConfusionMatrix(10, labels=[i for i in range(10)])
    total =  0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            cm.update(preds, labels.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    # 统计cm，获得缩放系数
    if mode == 'test':
        ratio,re_std = cm.summary(alpha)

        return 100 * correct / total, ratio,re_std
    else:
        return 100 * correct / total



def test_tsne(models, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'train'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    out_vec = torch.zeros(0)
    label = torch.zeros(0).long()
    with torch.no_grad():
        for (inputs, labels) in dataloaders:
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            preds = scores.cpu()
            labels = labels.cpu()
            out_vec = torch.cat([out_vec, preds])
            label = torch.cat([label, labels])
        out_vec = out_vec.numpy()
        label = label.numpy()
    return out_vec, label


iters = 0


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].long().cuda()
            index = data[2].detach().numpy().tolist()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, emb, features = models['backbone'](inputs)
        target_loss = criterion['CE'](scores, labels)
        probs = torch.softmax(scores, dim=1)
        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            moving_prob = data[3].cuda()
        moving_prob = (moving_prob * epoch + probs * 1) / (epoch + 1)
        dataloaders['train'].dataset.moving_prob[index, :] = moving_prob.cpu().detach().numpy()

        cumulative_logit = models['module'](features)
        m_module_loss = criterion['KL_Div'](F.log_softmax(cumulative_logit, 1), moving_prob.detach())
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss = m_backbone_loss + config.WEIGHT * m_module_loss



        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
    return loss



def train(cur_seed,cycle,models,criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    best_acc = 0.

    for epoch in range(num_epochs):

        # best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models,criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step()
        schedulers['module'].step()
        if False and epoch % 20 == 7:
            acc = test(models, dataloaders,1.0 ,mode='test')
            if best_acc < acc:
                best_acc = acc
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
