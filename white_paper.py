### Utilities
import os
import random 
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import json

### Torch
import torch
from torch import nn
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import MultiStepLR

### Custom
from src.model import model
from src.train import train, validate
from src.utils import Accuracy, save_checkpoint

parser = argparse.ArgumentParser(description='ResNet Training')
parser.add_argument('cfg', metavar='cfg', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

def main():

    best_acc = 0
    last_epoch = 0
    args = parser.parse_args()
    
    with open(args.cfg) as configurations:
        cfg, cfg_CIFAR, cfg_dataloader_train, cfg_dataloader_test, cfg_train = json.load(configurations).values()


    cfg["device"] = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")

    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
    trainset = CIFAR10(transform=train_transform, train=True, **cfg_CIFAR)
    trainloader = DataLoader(dataset=trainset, **cfg_dataloader_train)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    testset = CIFAR10(transform=test_transform, train=False, **cfg_CIFAR)
    testloader = DataLoader(dataset=testset, **cfg_dataloader_test)

    ResNet20 = model().to(cfg["device"])
    optimResNet20 = AdamW(ResNet20.parameters(), lr=1e-2)
    schedResNet20 = MultiStepLR(optimResNet20, last_epoch=-1,
                                milestones=[100, 150], gamma=0.1)

    if os.path.exists(cfg["checkpoint_path"]):
        checkpoint = torch.load(cfg["checkpoint_path"])
        last_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        ResNet20.load_state_dict(checkpoint["state_dict"])
        optimResNet20.load_state_dict(checkpoint["optimizer"])
        schedResNet20.load_state_dict(checkpoint["scheduler"])

    CELoss = nn.CrossEntropyLoss(reduction="sum")
    Acc = Accuracy(reduction="sum")

    if args.evaluate:
        ce, acc = validate(testloader, ResNet20, CELoss, Acc, verbose=True)
        print("Cross Entropy: {}, Accuracy: {}".format(ce.avg, acc.avg))
        return

    for epoch in (pbar := tqdm(range(last_epoch, cfg_train["n_epoches"]+last_epoch))):
        CE_train, acc_train = train(trainloader, ResNet20, optimResNet20, CELoss, Acc, cfg["device"])
        CE_test, acc_test = validate(trainloader, ResNet20, CELoss, Acc, cfg["device"])
        schedResNet20.step()

        is_best = acc_test.avg > best_acc
        if is_best:
            best_acc = acc_test.avg
            torch.save({
                'epoch': epoch + 1,
                'state_dict': ResNet20.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimResNet20.state_dict(),
                'scheduler' : schedResNet20.state_dict()
            }, cfg["checkpoint_path"])

        pbar.set_description("Best Acc.: {} | Train: CE {:.3f} Acc. {:.3f}| Test: CE {:.3f} Acc. {:.3f} | LR: {}".format(
                             best_acc, CE_train.avg, acc_train.avg, CE_test.avg, acc_test.avg, schedResNet20.get_last_lr()[0]))

if __name__ == '__main__':
    main()