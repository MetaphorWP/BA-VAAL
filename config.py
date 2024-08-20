import numpy as np

""" Configuration File.
"""

CUDA_VISIBLE_DEVICES = 1  # 设备
BATCH_SIZE = 128  # 批量大小

VAL_SUBSET = 10000

TRIALS = 5 # 训练轮次
CYCLES = 10  # 单次训练采样次数 + 1(初始化)
imb_factor = 1
# Random_Seed = [int(np.random.randint(1, 10000000)) for _ in range(TRIALS)]
cifar10_seed = [2432750,5567049,5603769,5569409,7928111]
svhn_seed = [8253471,5953742,1430772,1945368,1056831]
fashionmnist_seed = [2432750,5567049,5603769,5569409,7928111]

cinic_seed = [2432750,5567049,5603769,5569409,7928111]
DATASETS = ['cifar10', 'cifar10im', 'cinic10', 'fashionmnist', 'svhn']

WEIGHT = 1.0  # lambda
EPOCH = 200
EPOCH_GCN = 200
LearningRate = 1e-1
LR_GCN = 1e-3
MILESTONES = [160, 240]
EPOCHL = 120  # 20 #120 # After 120 epochs, stop
EPOCHV = 100  # VAAL number of epochs
MOMENTUM = 0.9
WDECAY = 5e-4  # 2e-3 # 5e-4
