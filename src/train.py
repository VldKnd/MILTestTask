### evaluate - evaluation cycle
### train - train cycle
from tqdm import tqdm
from src.utils import AverageMeter

def train(loader, model, optimizer, loss, criterion, device):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Accuracy')

    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        X_size = X_batch.size(0)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        output = loss(logits, y_batch)
        accuracy = criterion(logits, y_batch)
        
        output.backward()
        optimizer.step()

        losses.update(output.item(), X_size)
        top1.update(accuracy, X_size)
        
    return losses, top1

def validate(loader, model, loss, criterion, device, verbose=False):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Accuracy')

    model.eval()
    rng = tqdm(loader) if verbose else loader

    for X_batch, y_batch in rng:
        X_size = X_batch.size(0)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        output = loss(logits, y_batch)
        accuracy = criterion(logits, y_batch)

        losses.update(output.item(), X_size)
        top1.update(accuracy, X_size)
        
        
    return losses, top1
