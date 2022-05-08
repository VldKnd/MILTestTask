### Accuracy - calculates accuracy
### save_checkpoint - saves model
import torch

class Accuracy(object):

    def __init__(self, reduction="sum"):
        """
        Description
        """
        if reduction not in ["mean", "sum"]:
            raise AttributeError('The reduction can be either sum or mean')
            
        self.reduction = reduction
        
    @torch.no_grad()
    def __call__(self, x ,y):
        """
        Description
        """
        if self.reduction == "sum":
            return (x.argmax(1) == y).float().sum().item()
        else:
            return (x.argmax(1) == y).float().mean().item()

class AverageMeter(object):

    def __init__(self, name):
        """
        Description
        """
        self.name = name
        self.reset()

    def reset(self):
        """
        Description
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Description
        """
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count