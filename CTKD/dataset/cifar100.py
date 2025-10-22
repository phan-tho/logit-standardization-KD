from __future__ import print_function

import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.data import Subset


"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

mean = [0.5071, 0.4867, 0.4408]                                 
stdv = [0.2675, 0.2565, 0.2761]

def get_data_folder():
    """
    return the path to store the data
    """
    data_folder = 'your_cifar_data_path'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class CIFAR100BackCompat(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

class CIFAR100Instance(CIFAR100BackCompat):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def get_cifar10_dataloaders(batch_size=128, num_workers=8, imb_factor=1, n_omits=0):
    """
    cifar 10
    mean: 0.49139968, 0.48215827 ,0.44653124
    std: 0.24703233 0.24348505 0.26158768
    """
    mean = [0.49139968, 0.48215827, 0.44653124]
    stdv = [0.24703233, 0.24348505, 0.26158768]

    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    train_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=True,
                                 transform=train_transform)
    if n_omits > 0:
        train_set = omit_last_classes(train_set, n_omits)
    if imb_factor > 1:
        train_set = make_imbalanced_dataset(train_set, imb_factor)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR10(root=data_folder,
                                download=True,
                                train=False,
                                transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    
    # print len train set and test set
    print(f'Number of training samples: {len(train_set)}')
    print(f'Number of test samples: {len(test_set)}')

    return train_loader, test_loader

def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False, imb_factor=1, n_omits=0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    if n_omits > 0:
        train_set = omit_last_classes(train_set, n_omits)
    if imb_factor > 1:
        train_set = make_imbalanced_dataset(train_set, imb_factor)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    
    # print len train set and test set
    print(f'Number of training samples: {len(train_set)}')
    print(f'Number of test samples: {len(test_set)}')

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class CIFAR100InstanceSample(CIFAR100BackCompat):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets
       
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data


def make_imbalanced_dataset(dataset, imb_factor):
    rng = np.random.default_rng()
    targets = np.array([label for (_, label) in dataset])
    n_class = len(set(targets))
    cnt_cls = np.array([sum(targets == i) for i in range(n_class)])
    img_max = min(cnt_cls)

    imb_factor = 1/imb_factor

    img_num_per_cls = []
    for cls_idx in range(n_class):
        num = img_max * (imb_factor**(cls_idx / (n_class - 1.0)))
        img_num_per_cls.append(int(num))

    indices_list = []
    for cls_idx, num_samples in enumerate(img_num_per_cls):
        cls_indices = np.where(targets == cls_idx)[0]
        sampled_indices = rng.choice(cls_indices, size=num_samples, replace=False)
        indices_list.append(sampled_indices)

    all_indices = np.concatenate(indices_list)
    return Subset(dataset, all_indices)


def omit_last_classes(dataset, n_omits):
    targets = np.array([label for (_, label) in dataset])
    max_class = max(targets) + 1
    keep_classes = set(range(max_class - n_omits))

    keep_indices = [i for i, t in enumerate(targets) if t in keep_classes]
    return Subset(dataset, keep_indices)