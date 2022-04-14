import torch
import torch.nn as nn
import torch.optim as optim
import snntorch.functional as SF
import snntorch.spikegen as SGen

from typing import Optional

from tqdm import tqdm
from torch.utils.data import DataLoader


def _get_acc(outputs: torch.Tensor, targets: torch.Tensor, is_snn: bool):
    if is_snn:
        acc = SF.accuracy_rate(outputs, targets)
    else:
        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean()
    return acc


def standard_training(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    args
):
    train_acc = 0
    train_loss = 0
    device = torch.device(args.device)
    model.train()

    bid = 0
    prog = tqdm(dataloader)
    for data, targets in prog:
        bid += 1
        data = data.to(device)
        targets = targets.to(device)

        if args.rate_encoding:
            data = SGen.rate(data, args.time_steps)

        optimizer.zero_grad()

        # flatten input for linear layers
        if args.model in ['MLP', 'Linear']:
            if args.rate_encoding:
                data = data.view(args.time_steps, -1, 28 * 28)
            else:
                data = data.view(-1, 28 * 28)

        # forward
        if args.snn:
            outputs, mems = model(data, args.rate_encoding)
        else:
            outputs = model(data)

        # compute loss
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # accuracy
        acc = _get_acc(outputs, targets, args.snn)

        train_acc += acc.item()
        train_loss += loss.item()

        avg_acc = train_acc / bid
        avg_loss = train_loss / bid

        prog.set_description(
            f'| Epoch {epoch} | Acc {avg_acc:.4f} | Loss {avg_loss:.4f} |')

    return avg_acc, avg_loss


def random_sampling_training(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    args
):
    with torch.no_grad():
        train_acc = 0
        train_loss = 0
        device = torch.device(args.device)
        model.eval()

        bid = 0
        prog = tqdm(dataloader)
        for data, targets in prog:
            bid += 1
            data = data.to(device)
            targets = targets.to(device)

            if args.rate_encoding:
                data = SGen.rate(data, args.time_steps)

            if args.model in ['MLP', 'Linear']:
                if args.rate_encoding:
                    data = data.view(args.time_steps, -1, 28 * 28)
                else:
                    data = data.view(-1, 28 * 28)

            # forward
            if args.snn:
                outputs, mems = model(data)
            else:
                outputs = model(data)

            # compute loss
            loss = loss_fn(outputs, targets)

            # RANDOM SEARCH update
            for _ in range(args.random_search_steps):
                # generate purturbations
                state_dict_save = model.state_dict()
                for name, param in model.named_parameters():
                    perturb = torch.randn_like(param)
                    param.data = param.data + perturb * args.perturb_step

                # try computing loss
                if args.snn:
                    output_, mem_ = model(data, args.rate_encoding)
                else:
                    output_ = model(data)

                loss_attempt = loss_fn(output_, targets)
                if loss_attempt < loss:
                    # accept update if loss is lower
                    break
                else:
                    # restore model otherwise
                    model.load_state_dict(state_dict_save)

            # accuracy
            acc = _get_acc(outputs, targets, args.snn)

            train_acc += acc.item()
            train_loss += loss.item()

            avg_acc = train_acc / bid
            avg_loss = train_loss / bid

            prog.set_description(
                f'| Epoch {epoch} | Acc {avg_acc:.4f} | Loss {avg_loss:.4f} |')

    return avg_acc, avg_loss


def spsa_training(
    epoch: int,
    model: nn.Module,
    model_p: nn.Module,
    model_m: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    args,
    optimizer: Optional[optim.Optimizer] = None
):
    with torch.no_grad():
        train_acc = 0
        train_loss = 0
        device = torch.device(args.device)
        model.eval()

        bid = 0
        prog = tqdm(dataloader)
        for data, targets in prog:
            bid += 1
            data = data.to(device)
            targets = targets.to(device)

            if args.rate_encoding:
                data = SGen.rate(data, args.time_steps)

            if args.model in ['MLP', 'Linear']:
                if args.rate_encoding:
                    data = data.view(args.time_steps, -1, 28 * 28)
                else:
                    data = data.view(-1, 28 * 28)

            # forward
            if args.snn:
                outputs, mems = model(data, args.rate_encoding)
            else:
                outputs = model(data)

            # compute loss
            loss = loss_fn(outputs, targets)

            # SPSA Update
            # reset params
            model_p.load_state_dict(model.state_dict())
            model_m.load_state_dict(model.state_dict())

            # generate purturbations
            perturb_dict = {}
            for name, param in model.named_parameters():
                perturb = torch.sign(torch.randn_like(param))
                perturb_dict[name] = perturb * args.perturb_step

            # update
            for name, param in model_p.named_parameters():
                param.data = param.data + perturb_dict[name]
            for name, param in model_m.named_parameters():
                param.data = param.data - perturb_dict[name]

            # forward pass
            if args.snn:
                output_p, mem_p = model_p(data, args.rate_encoding)
                output_m, mem_m = model_m(data, args.rate_encoding)
            else:
                output_p = model_p(data)
                output_m = model_m(data)

            # compute loss
            loss_p = loss_fn(output_p, targets)
            loss_m = loss_fn(output_m, targets)

            # estimated gradient descent
            grad_pred = (loss_p - loss_m) / (2 * args.perturb_step)
            if optimizer is not None:
                optimizer.zero_grad()
                for name, param in model.named_parameters():
                    param.grad = grad_pred * perturb_dict[name]
                optimizer.step()
            else:
                for name, param in model.named_parameters():
                    update = args.lr * grad_pred * perturb_dict[name]
                    param.data = param.data - update

            # accuracy
            acc = _get_acc(outputs, targets, args.snn)

            train_acc += acc.item()
            train_loss += loss.item()

            avg_acc = train_acc / bid
            avg_loss = train_loss / bid

            prog.set_description(
                f'| Epoch {epoch} | Acc {avg_acc:.4f} | Loss {avg_loss:.4f} |')
    return avg_acc, avg_loss


def evaluation(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    args
):
    print(f'| Starting Evaluation for epoch {epoch} |')
    device = torch.device(args.device)
    with torch.no_grad():
        test_acc = 0
        tot_samples = 0
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            if args.rate_encoding:
                data = SGen.rate(data, args.time_steps)

            if args.model in ['MLP', 'Linear']:
                if args.rate_encoding:
                    data = data.view(args.time_steps, -1, 28 * 28)
                else:
                    data = data.view(-1, 28 * 28)

            if args.snn:
                output, mem = model(data, args.rate_encoding)
            else:
                output = model(data)

            acc = _get_acc(output, targets, args.snn) * targets.shape[0]
            test_acc += acc.item()
            tot_samples += targets.shape[0]

        avg_acc = test_acc / tot_samples
        print(f'| Epoch {epoch} | Test Acc {avg_acc:.4f} |')

    return avg_acc
