# Python
import os
import random
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

# Custom
from utils.check_category import get_category
import models.task_model as task_model
from models.query_model import TDNet
from train_test_task_ranker import train, test
from load_dataset import load_dataset
from selection_methods import update_fake_unlabeled_dataset, query_samples_Confid,query_samples
import config

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="cifar10",
                    help="cifar10 / svhn")
parser.add_argument("--init_dist", type=str, default='random',
                    help="uniform / random.")
parser.add_argument("-w", "--num_workers", type=str, default=0,
                    help="The number of workers.")
parser.add_argument("-e", "--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-c", "--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-b", "--alpha", type=float, default=0.5,
                    help=" preds ratio lambda = alpha * 2")

args = parser.parse_args()

args.add_num = {
    'cifar10': 1000,
    'cifar10im': 1000,
    'svhn': 500,
    'fashionmnist': 1000,
    'cinic10': 1000
}[args.dataset]

args.subset = 10000

args.initial_size = 1000
##
# Main
if __name__ == '__main__':

    assert args.dataset in config.DATASETS, 'No dataset %s! Try options %s' % (args.dataset, config.DATASETS)

    time.asctime()
    os.makedirs('results', exist_ok=True)
    txt_name = f'results/results_{args.dataset}_{args.initial_size}_BA-TiVAAL_Lambda{args.alpha * 2}_newRatio_onlyConf2.txt'
    results = open(txt_name, 'w')
    results.write(str(time.asctime()))
    results.write("\n")
    print("Current Dataset: %s" % args.dataset)
    # 若使用全部数据训练，则轮次为1，采样为0
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles
        TRIALS = config.TRIALS
    for trial in range(TRIALS):

        cur_seed = config.cifar10_seed[trial]
        args.seed = cur_seed
        print("cur_seed is :{}".format(cur_seed))

        # 设置随机种子
        torch.manual_seed(cur_seed)
        torch.cuda.manual_seed_all(cur_seed)
        np.random.seed(cur_seed)
        random.seed(cur_seed)
        torch.backends.cudnn.deterministic = True

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, increment, NO_CLASSES, NUM_TRAIN = load_dataset(args)
        print('The entire datasize is {}'.format(len(data_train)))
        # ADDENDUM = increment
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set = indices
        else:
            labeled_set = indices[:args.initial_size]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        # 使用已标注数据构建loader
        train_loader = DataLoader(data_train, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True, drop_last=False)

        # 获取当前待分类数据(已标注数据)的类别分布
        freq = get_category(train_loader)
        print(freq)

        # 构建测试loader
        test_loader = DataLoader(data_test, batch_size=config.BATCH_SIZE)
        dataloaders = {'train': train_loader, 'test': test_loader}

        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            # resnet18    = vgg11().cuda()
            if args.dataset != 'fashionmnist':
                resnet18 = task_model.ResNet18(num_classes=NO_CLASSES).cuda()
                loss_module = TDNet(out_dim=NO_CLASSES).cuda()
            else:
                resnet18 = task_model.ResNet18fm(num_classes=NO_CLASSES).cuda()
                loss_module = TDNet(feature_sizes=[28, 14, 7, 4], out_dim=NO_CLASSES).cuda()

        models = {'backbone': resnet18, 'module': loss_module}
        # torch.backends.cudnn.benchmark = True

        # Loss, criterion and scheduler (re)initialization
        criterion = {}
        criterion['CE'] = nn.CrossEntropyLoss(reduction='none')
        criterion['KL_Div'] = nn.KLDivLoss(reduction='batchmean')
        for cycle in range(CYCLES):

            if not args.total:
                # 随机选择SUBSET数量未标记数据点索引构建subset
                random.shuffle(unlabeled_set)
                unlabeled_subset = unlabeled_set[:args.subset]

            torch.backends.cudnn.benchmark = False
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=config.LearningRate,
                                       momentum=config.MOMENTUM, weight_decay=config.WDECAY)

            scheduler_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=config.MILESTONES)

            optim_module = optim.SGD(models['module'].parameters(), lr=config.LearningRate,
                                     momentum=config.MOMENTUM, weight_decay=config.WDECAY)
            scheduler_module = lr_scheduler.MultiStepLR(optim_module, milestones=config.MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': scheduler_backbone, 'module': scheduler_module}

            # Training and testing
            train(cur_seed, cycle, models, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs,
                  config.EPOCHL)

            acc, ratio, re_std = test(models, dataloaders, args.alpha, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))
            np.array([trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc, re_std, re_std * 100 / acc]).tofile(
                results, sep=" ")
            results.write("\n")

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            # 得到一未标记数据的dataloader，称为fake_unlabeled_loader

            fake_unlabeled_loader = DataLoader(data_unlabeled, batch_size=config.BATCH_SIZE,
                                               sampler=SubsetRandomSampler(unlabeled_subset), drop_last=False)
            update_fake_unlabeled_dataset(models, fake_unlabeled_loader)

            # Get the indices of the unlabeled samples to train on next cycle
            # arg = query_samples(models, data_unlabeled, unlabeled_subset, labeled_set, cycle, args, ratio)
            arg = query_samples_Confid(models, data_unlabeled, unlabeled_subset, labeled_set, cycle, args, ratio)

            # Update the labeled dataset and the unlabeled dataset, respectively
            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(unlabeled_subset)[arg][-increment:].numpy())
            listd = list(torch.tensor(unlabeled_subset)[arg][:-increment].numpy())
            unlabeled_set = listd + unlabeled_set[args.subset:]
            # print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=config.BATCH_SIZE,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            cur_freq = get_category(dataloaders['train'])
            print(cur_freq)
    time.asctime()
    results.write(str(time.asctime()))
    results.close()
