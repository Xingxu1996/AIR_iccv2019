import torch
import numpy as np


def fit(train_loader, val_loader, embedding_net, model, loss1_fn, loss2_fn, optimizer, scheduler, n_epochs, cuda, log_interval,metrics=[],start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss1_fn, loss2_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        # for metric in metrics1:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, val_loss1, val_loss2, metrics = test_epoch(val_loader, model, loss1_fn, loss2_fn, cuda, metrics)
        val_loss /= len(val_loader)
        val_loss1 /= len(val_loader)
        val_loss2 /= len(val_loader)

        message = '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}.loss1:{:.4f}.loss2:{:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss, val_loss1, val_loss2)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print(message)
        if epoch % 21 == 0:
            #torch.save(embedding_net.state_dict(), 'shiyan.pkl')
            torch.save(model.state_dict(), 'new24.pkl')

def train_epoch(train_loader, model, loss1_fn, loss2_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()
    # for metric in metrics1:
    #     metric.reset()

    model.train()
    losses = []
    losses0 = []
    losses1 = []
    losses2 = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        target2 = target2 if len(target2) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
            if target2 is not None:
                target2 = target2.cuda()


        optimizer.zero_grad()

        c2, c4, outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        # if type(c1) not in (tuple, list):
        #     c1 = (c1,)
        if type(c2) not in (tuple, list):
            c2 = (c2,)

        if type(c4) not in (tuple, list):
            c4 = (c4,)

        if target is not None:
            target = (target,)
        if target2 is not None:
            target2 = (target2,)

        loss0_outputs = loss1_fn(c2[0], target2[0])
        loss1_outputs = loss1_fn(c4[0], target[0])
        loss2_outputs = loss2_fn(outputs[0], target[0], c2[0], c4[0])

        loss0 = loss0_outputs[0] if type(loss0_outputs) in (tuple, list) else loss0_outputs
        loss1 = loss1_outputs[0] if type(loss1_outputs) in (tuple, list) else loss1_outputs
        loss2 = loss2_outputs[0] if type(loss2_outputs) in (tuple, list) else loss2_outputs

        loss = 0.5 * (loss1+loss0) + 0.5 * loss2
        losses.append(loss.item())
        losses0.append(loss0.item())
        losses1.append(loss1.item())
        losses2.append(loss2.item())

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(c4, target, loss1_outputs)


        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses),np.mean(losses1),np.mean(losses2))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss1_fn, loss2_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        val_loss0 = 0
        val_loss1 = 0
        val_loss2 = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            target2 = target2 if len(target2) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                if target2 is not None:
                    target2 = target2.cuda()

            c2, c4, outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            if type(c2) not in (tuple, list):
                c2 = (c2,)

            if type(c4) not in (tuple, list):
                c4 = (c4,)

            if target is not None:
                target = (target,)
            if target2 is not None:
                target2 = (target2,)

            loss0_outputs = loss1_fn(c2[0], target2[0])
            loss1_outputs = loss1_fn(c4[0], target[0])
            loss2_outputs = loss2_fn(outputs[0], target[0], c2[0], c4[0])

            loss0 = loss0_outputs[0] if type(loss0_outputs) in (tuple, list) else loss0_outputs
            loss1 = loss1_outputs[0] if type(loss1_outputs) in (tuple, list) else loss1_outputs
            loss2 = loss2_outputs[0] if type(loss2_outputs) in (tuple, list) else loss2_outputs

            loss = 0.5*(loss0 + loss1) + 0.5 * loss2

            val_loss += loss.item()
            val_loss0 += loss0.item()
            val_loss1 += loss1.item()
            val_loss2 += loss2.item()

            for metric in metrics:
                metric(c4, target, loss1_outputs)

    return val_loss, val_loss1, val_loss2, metrics
