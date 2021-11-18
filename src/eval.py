import torch


@torch.no_grad()
def eval_accuracy(ctx, model, testloader, desc=None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, correct, total = 0, 0, 0
    with ctx.tqdm(testloader, desc=desc, leave=True) as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(ctx.device), targets.to(ctx.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total
    return acc

