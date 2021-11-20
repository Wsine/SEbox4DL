import torch
from tqdm import tqdm

from src.dataset import load_dataset
from src.model import load_model
from src.arguments import commparser as parser
from src.utils import *


def train(ctx, model, trainloader, optimizer, criterion, device, desc):
    model.train()
    train_loss, correct, total = 0, 0, 0
    with ctx.tqdm(trainloader, desc=desc) as tepoch:
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

            avg_loss = train_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    return acc, avg_loss

def train_model(ctx, model, trainloader, validloader, optimizer, criterion, scheduler):
    # set as default
    ctx.opt.parallel = False
    if ctx.device.type == 'cuda':
        if torch.cuda.device_count() > 1 and ctx.opt.parallel is True:
            model = torch.nn.DataParallel(model)
    guard_folder(ctx.opt)
    start_epoch = -1
    best_acc = 0
    for epoch in ctx.tqdm(range(start_epoch + 1, ctx.opt.max_epoch), desc="Total Progress"):
        sub_desc = 'Epoch {}'.format(epoch)
        train(ctx, model, trainloader, optimizer, criterion, ctx.device, sub_desc)
        acc, _ = test(ctx, model, validloader, criterion, ctx.device)
        if acc > best_acc:
            # print('Saving model')
            state = {
                'epoch': epoch,
                'net': model.state_dict() if not ctx.opt.parallel else model.module.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            if ctx.opt.pretrained:
                state = 'pretrained'
            else:
                state = 'train'
            torch.save(state, get_model_path(ctx.opt, state=state))
            best_acc = acc
        scheduler.step()

@torch.no_grad()
def test(
        ctx,
        model, valloader, criterion, device,
        desc='Evaluate', return_label=False, tqdm_leave=True):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    pred_labels, trg_labels = [], []
    with ctx.tqdm(valloader, desc=desc, leave=tqdm_leave) as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if return_label:
                pred_labels.append(predicted.cpu())
                trg_labels.append(targets.cpu())

            avg_loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total

    if return_label:
        pred_labels = torch.cat(pred_labels)
        trg_labels = torch.cat(trg_labels)
        return acc, (pred_labels, trg_labels)

    return acc, (None, None)


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    _, trainloader = load_dataset(opt, split='train')
    _, testloader = load_dataset(opt, split='test')
    model = load_model(opt)
    opt.parallel = False
    if device.type == 'cuda':
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            opt.parallel = True
        else:
            device = torch.device(f'cuda:{opt.gpu}')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1
    best_acc = 0
    if opt.resume:
        ckp = torch.load(get_model_path(opt, state='pretrained'))
        if opt.parallel:
            model.module.load_state_dict(ckp['net'])
        else:
            model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['sched'])
        start_epoch = ckp['epoch']
        best_acc = ckp['acc']

    for epoch in range(start_epoch + 1, opt.max_epoch):
        print('Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, _ = test(model, testloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'epoch': epoch,
                'net': model.state_dict() if not opt.parallel else model.module.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state='pretrained'))
            best_acc = acc
        scheduler.step()


if __name__ == '__main__':
    main()

