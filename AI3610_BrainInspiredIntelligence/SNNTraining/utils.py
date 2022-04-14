import models.ann
import models.snn

import snntorch.surrogate as surrogate


def _get_ann(args):
    if args.model == 'Linear':
        return models.ann.ANNLinear(28 * 28, 10)
    elif args.model == 'MLP':
        return models.ann.ANNMLP(28 * 28, 256, 10)
    elif args.model == 'Conv':
        return models.ann.ANNConvNet()
    else:
        raise ValueError(f'Unknown model: {args.model}')


def _get_snn(args):
    if args.model == 'Linear':
        return models.snn.SNNLinear(
            28 * 28, 10,
            beta=args.beta,
            spike_grad=surrogate.fast_sigmoid(),
            steps=args.time_steps)
    elif args.model == 'MLP':
        return models.snn.SNNMLP(
            28 * 28, 256, 10,
            beta=args.beta,
            spike_grad=surrogate.fast_sigmoid(),
            steps=args.time_steps)
    elif args.model == 'Conv':
        return models.snn.SNNConvNet(
            beta=args.beta,
            spike_grad=surrogate.fast_sigmoid(),
            steps=args.time_steps)
    else:
        raise ValueError(f'Unknown model: {args.model}')


def get_model(args):
    if args.snn:
        return _get_snn(args)
    else:
        return _get_ann(args)