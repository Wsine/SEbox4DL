import torch

from src.utils import *


def train(ctx, model, trainloader, optimizer, criterion, device):
    model.train()
    train_loss, num_batches = 0, 0
    correct, total = 0, 0
    with ctx.tqdm(trainloader, desc='Train') as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            num_batches = batch_idx + 1
            avg_loss = train_loss / num_batches
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total
    avg_loss = train_loss / num_batches
    return acc, avg_loss


@torch.no_grad()
def test(ctx, model, valloader, criterion, device):
    model.eval()

    test_loss, num_batches = 0, 0
    correct, total = 0, 0
    with ctx.tqdm(valloader, desc='Test', leave=False) as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            num_batches = batch_idx + 1
            avg_loss = test_loss / num_batches
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total
    avg_loss = test_loss / num_batches
    return acc, avg_loss


def train_model(ctx, model, trainloader, validloader, is_finetune=False):
    guard_folder(ctx)

    criterion = eval(f'torch.nn.{ctx.opt.criterion}()')
    optimizer = eval(f'torch.optim.{ctx.opt.optimizer}')(
        model.parameters(),
        lr=ctx.opt.lr, momentum=ctx.opt.momentum, weight_decay=ctx.opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, best_loss = 0, 0
    for epoch in ctx.tqdm(range(0, ctx.opt.max_epoch), desc='Epochs'):
        train(ctx, model, trainloader, optimizer, criterion, ctx.device)
        acc, loss = test(ctx, model, validloader, criterion, ctx.device)
        if acc > best_acc:
            # print('Saving model')
            model_state = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc,
                'loss': loss
            }
            state = 'best' if is_finetune is False else 'finetune'
            torch.save(model_state, get_model_path(ctx, state=state))
            best_acc, best_loss = acc, loss
        scheduler.step()

    return best_acc, best_loss

