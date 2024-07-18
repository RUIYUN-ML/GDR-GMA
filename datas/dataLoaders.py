import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datas.TinyImageNet import TinyImageNet
from seed import set_seed
from torch.utils.data import DataLoader
import random

class DataLoaders:
    def __init__(self, dataset, batch_size=256, unlearned_size=0.2, seed=0):
        self.data_dir = "./data"
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.unlearned_size = unlearned_size
        self.data_transforms = {
            "cifar-train": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            "cifar-val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            "imagenet-train": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
            "imagenet-val": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
        }

    def load_data(self):
        if self.dataset == "cifar-10":
            data_train = datasets.CIFAR10(
                root=self.data_dir,
                transform=self.data_transforms["cifar-train"],
                train=True,
                download=True,
            )

            data_val = datasets.CIFAR10(
                root=self.data_dir,
                transform=self.data_transforms["cifar-val"],
                train=False,
                download=True,
            )
        elif self.dataset == "imagenet":
            data_train = TinyImageNet(
                self.data_dir + "tiny-imagenet-200",
                train=True,
                transform=self.data_transforms["imagenet-train"],
            )
            data_val = TinyImageNet(
                self.data_dir + "tiny-imagenet-200",
                train=False,
                transform=self.data_transforms["imagenet-val"],
            )

        image_datasets = {"train": data_train, "val": data_val}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders["train"] = DataLoader(
            image_datasets["train"],
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,

            num_workers=16,
        )
        dataloaders["val"] = DataLoader(
            image_datasets["val"],
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,

            num_workers=16,
        )

        return dataloaders
    
    def load_unlearn_data(self):  
        if self.dataset == "cifar-10":
            data_train = datasets.CIFAR10(
                root=self.data_dir,
                transform=self.data_transforms["cifar-train"],
                train=True,
                download=True,
            )

            data_val = datasets.CIFAR10(
                root=self.data_dir,
                transform=self.data_transforms["cifar-val"],
                train=False,
                download=True,
            )
        elif self.dataset == "imagenet":
            data_train = TinyImageNet(
                self.data_dir + "tiny-imagenet-200",
                train=True,
                transform=self.data_transforms["imagenet-train"],
            )
            data_val = TinyImageNet(
                self.data_dir + "tiny-imagenet-200",
                train=False,
                transform=self.data_transforms["imagenet-val"],
            )
        
        target_dataset, non_target_dataset = torch.utils.data.random_split(data_train, [int(len(data_train)*self.unlearned_size), int(len(data_train)-int(len(data_train)*self.unlearned_size))])


        dataloaders = {}   
        dataloaders["remain"] = DataLoader(
            non_target_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=8,
        )
        dataloaders["unlearn"] = DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=8,
        )
        dataloaders["val"] = DataLoader(
            data_val,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
        )
        
        return dataloaders

            
        