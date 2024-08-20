import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, SVHN, FashionMNIST, ImageFolder

from config import *
from utils.make_imbalance import *


class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transfer=None, args=None):

        self.dataset_name = dataset_name
        self.args = args

        if args is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True

        if self.dataset_name == "cifar10":
            self.dataset = CIFAR10('data/cifar10', train=train_flag,
                                   download=True, transform=transfer)
        if self.dataset_name == "svhn":
            self.dataset = SVHN('data/svhn', split="train",
                                download=True, transform=transfer)
        if self.dataset_name == "fashionmnist":
            self.dataset = FashionMNIST('../fashionMNIST', train=train_flag,
                                        download=True, transform=transfer)
        if self.dataset_name == 'cinic10':
            path = r'data/cinic10/CINIC-10'
            if train_flag:
                self.dataset = torchvision.datasets.ImageFolder(path + r'/train', transform=transfer)
            else:
                self.dataset = torchvision.datasets.ImageFolder(path + r'/test', transform=transfer)

        if self.dataset_name == "cifar10im":
            self.dataset = CIFAR10('data/cifar10', train=train_flag,
                                   download=True, transform=transfer)

            imbalance_class_counts = [int(5000 / imb_factor), 5000] * 5
            targets = np.array(self.dataset.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
            # Get imbalanced number of instances
            imbalance_class_indices = [class_idx[:class_count] for class_idx, class_count in
                                       zip(class_indices, imbalance_class_counts)]
            imbalance_class_indices = np.hstack(imbalance_class_indices)

            # Set target and data to dataset
            self.dataset.targets = targets[imbalance_class_indices]
            self.dataset.data = self.dataset.data[imbalance_class_indices]

        if (self.dataset_name in ["cifar10", "cifar10im", "fashionmnist", "cinic10"]) and (args is not None):
            targets = np.array(self.dataset.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            self.moving_prob = np.zeros((len(self.dataset), nb_classes), dtype=np.float32)

        elif (self.dataset_name == "svhn") and (args is not None):
            labels = np.array(self.dataset.labels)
            classes, class_counts = np.unique(labels, return_counts=True)
            nb_classes = len(classes)
            self.moving_prob = np.zeros((len(self.dataset), nb_classes), dtype=np.float32)

    def __getitem__(self, index):
        if self.args is not None:
            data, target = self.dataset[index]
            moving_prob = self.moving_prob[index]
            return data, target, index, moving_prob
        else:
            data, target = self.dataset[index]
            return data, target, index

    def __len__(self):
        return len(self.dataset)


def transformer(mean, std):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize(mean, std)
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    return [train_transform, test_transform]


def load_dataset(args):
    dataset = args.dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    if dataset == 'cifar10':
        train_transform, test_transform = transformer(cifar10_mean, cifar10_std)
        data_train = MyDataset(dataset, train_flag=True, transfer=train_transform, args=args)
        data_unlabeled = MyDataset(dataset, train_flag=True, transfer=test_transform)

        data_test = CIFAR10('data/cifar10', train=False, download=True, transform=test_transform)

        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN

    elif dataset == 'cifar10im':
        train_transform, test_transform = transformer(cifar10_mean, cifar10_std)
        data_train = MyDataset(dataset, train_flag=True, transfer=train_transform, args=args)
        data_unlabeled = MyDataset(dataset, train_flag=True, transfer=test_transform)

        data_test = CIFAR10('data/cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN

    elif dataset == "fashionmnist":
        data_train = MyDataset(dataset, train_flag=True, transfer=T.Compose([T.RandomHorizontalFlip(),
                                                                             T.RandomCrop(28, padding=2),
                                                                             T.ToTensor(),
                                                                             T.Normalize([0.286], [0.352])]), args=args)

        data_unlabeled = MyDataset(dataset, train_flag=True, transfer=T.Compose([
            T.ToTensor(),
            T.Normalize([0.286], [0.352])]))

        data_test = FashionMNIST('../fashionmnist', train=False, transform=T.Compose([
            T.ToTensor(),
            T.Normalize([0.286], [0.352])]), download=True)
        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN

    elif dataset == 'svhn':
        data_train = MyDataset(dataset, train_flag=True, transfer=T.Compose([T.ToTensor()]), args=args)
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))

        data_test = SVHN('data/svhn', split='test', download=True,
                         transform=T.Compose([T.ToTensor()]))

        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN
    elif dataset == 'cinic10':
        train_transform, test_transform = transformer(cinic_mean, cinic_std)
        data_train = MyDataset(dataset, train_flag=True, transfer=train_transform, args=args)
        data_unlabeled = MyDataset(dataset, train_flag=True, transfer=test_transform)

        data_test = torchvision.datasets.ImageFolder(r'data/cinic10/CINIC-10/test', transform=test_transform)

        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN
    else:
        raise RuntimeError()

    adden = args.add_num
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train
