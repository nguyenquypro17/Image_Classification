import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

def get_cifar10_loaders(batch_size=128, augmentation='none', data_dir='./data'):
    """augmentation: none, standard
    data_dir: thư mục lưu data
    Returns: train_loader (45k), val_loader (5k), test_loader (10k)"""
    
    if augmentation == 'standard':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # Default to 'none' if augmentation type not recognized
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    # Split train set: 45k train + 5k validation
    train_set, val_set = random_split(train_set, [45000, 5000])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader