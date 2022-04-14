import time
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import snntorch.functional as SF

import trainer
from args import parse_args, print_args
from utils import get_model
from data_prep import load_mnist, wrap_dataloader


def main():
    args = parse_args()
    print_args(args)

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = get_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # load pretrained model, if required
    if args.pretrained_init != '':
        state_dict = torch.load(args.pretrained_init)
        for name, params in model.named_parameters():
            if name in state_dict:
                params.data = state_dict[name]
                print(f'Loaded {name}')

    train_set, test_set = load_mnist(args.data_path)
    train_loader = wrap_dataloader(train_set, args.batch_size)
    test_loader = wrap_dataloader(test_set, args.test_batch_size)

    model_p = None
    model_m = None
    if args.trainer == 'SPSA':
        model_p = get_model(args).to(device)
        model_m = get_model(args).to(device)

    if args.snn:
        if args.loss == 'ce':
            loss_fn = SF.ce_count_loss()
        else:
            loss_fn = SF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25)
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_accs = []
    train_losses = []
    test_accs = []
    best_epoch = 0
    best_acc = 0
    best_dict = None

    start_time = time.time()
    for e in range(args.epochs):

        # training
        if not args.dry_run:
            if args.trainer == 'ADAM':
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                avg_acc, avg_loss = trainer.standard_training(
                    epoch=e,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    args=args)
            elif args.trainer == 'RANDOM':
                avg_acc, avg_loss = trainer.random_sampling_training(
                    epoch=e,
                    model=model,
                    loss_fn=loss_fn,
                    dataloader=train_loader,
                    args=args)
            elif args.trainer == 'SPSA':
                optimizer = (
                    optim.Adam(model.parameters(), lr=args.lr)
                    if args.spsa_use_adam else None)
                avg_acc, avg_loss = trainer.spsa_training(
                    epoch=e,
                    model=model,
                    model_p=model_p,
                    model_m=model_m,
                    loss_fn=loss_fn,
                    dataloader=train_loader,
                    args=args,
                    optimizer=optimizer)
            else:
                raise ValueError(f'Unknown trainer: {args.trainer}')

            train_accs.append(avg_acc)
            train_losses.append(avg_loss)

        # evaluation
        test_acc = trainer.evaluation(
            epoch=e,
            model=model,
            dataloader=test_loader,
            args=args)

        test_accs.append(test_acc)

        if test_acc > best_acc:
            best_dict = model.state_dict()
            best_acc = test_acc
            best_epoch = e

        if args.dry_run:
            break

    time_elapsed = time.time() - start_time

    print_args(args)
    print(f'Run completed in {time_elapsed:.2f} secs')
    print(f'Model parameters: {n_params}')
    print(f'Best test accuracy: {best_acc:.4f} at epoch {best_epoch}')

    # save results
    if not args.dry_run:
        snn_identifier = '-snn' if args.snn else ''
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        trace = {
            'train_accs': train_accs,
            'train_losses': train_losses,
            'test_accs': test_accs,
        }
        trace_name = f'{args.model}-{args.trainer}{snn_identifier}-{timestamp}.pkl'
        trace_path = f'./traces/{trace_name}'
        pickle.dump(trace, open(trace_path, 'wb'))
        print('Training stats saved as', trace_name)

        torch.save(best_dict, f'./saved_models/{args.model_save}')


if __name__ == '__main__':
    main()
