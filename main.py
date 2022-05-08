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
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

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
    trainset = CIFAR10(**cfg_CIFAR, transform=train_transform, train=True)
    trainloader = DataLoader(**cfg_dataloader_train, dataset=trainset)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    testset = CIFAR10(**cfg_CIFAR, transform=test_transform, train=False)
    testloader = DataLoader(**cfg_dataloader_test, dataset=testset)

    ResNet20 = model().to(cfg["device"])
    optimResNet20 = AdamW(ResNet20.parameters(), lr=1e-2)
    schedResNet20 = MultiStepLR(optimResNet20, last_epoch=-1,
                                milestones=[100, 150], gamma=0.1)

    if os.path.exists(cfg["checkpoint_path"]):
        checkpoint = torch.load(cfg["checkpoint_path"])
        last_epoch = checkpoint["epoch"] + 1
        ResNet20.load_state_dict(checkpoint["state_dict"])
        optimResNet20.load_state_dict(checkpoint["optimizer"])
        schedResNet20.load_state_dict(checkpoint["scheduler"])

    CELoss = nn.CrossEntropyLoss(reduction="sum")
    Acc = Accuracy(reduction="sum")

    if args.evaluate:
        validate(testloader, ResNet20, Acc)
        return

    for epoch in (pbar := tqdm(range(last_epoch, last_epoch+cfg_train["n_epoches"]))):
        CE_train, acc_train = train(trainloader, ResNet20, optimResNet20, schedResNet20, CELoss, Acc, cfg["device"])
        CE_test, acc_test = validate(trainloader, ResNet20, CELoss, Acc, cfg["device"])
        schedResNet20.step()

        is_best = acc_test > best_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': ResNet20.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimResNet20.state_dict(),
                'scheduler' : schedResNet20.state_dict()
            }, is_best, cfg["checkpoint_path"])

        pbar.set_description("Best Acc.: {} | Train: CE {:.3f} Acc. {:.3f}| Test: CE {:.3f} Acc. {:.3f} | LR: {}".format(
            best_acc, CE_train, acc_train, CE_test, acc_test, schedResNet20.get_last_lr()[0]
        ))

if __name__ == '__main__':
    main()