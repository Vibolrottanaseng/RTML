import argparse

from evaluate import linear_eval_dino, linear_eval_mae, linear_eval_simclr, visualize_mae_reconstruction
from train import train_dino, train_mae, train_simclr
from utils import ensure_dirs, get_device, save_stats, set_seed


def make_default_save_path(args):
    if args.model == 'dino':
        suffix = []
        if args.no_centering:
            suffix.append('no_centering')
        if args.n_local == 0:
            suffix.append('no_local')
        tag = '_' + '_'.join(suffix) if suffix else ''
        return f'saved/dino{tag}.pt'
    if args.model == 'mae':
        mask_tag = str(args.mask_ratio).replace('.', '')
        return f'saved/mae_encoder_mask{mask_tag}.pt'
    return 'saved/simclr.pt'


def main():
    parser = argparse.ArgumentParser(description='A3 Self-Supervised Learning: SimCLR, DINO, MAE')
    parser.add_argument('--model', choices=['simclr', 'dino', 'mae'], required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--reconstruct', action='store_true', help='Show MAE reconstruction grid')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--linear-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--n-local', type=int, default=4)
    parser.add_argument('--no-centering', action='store_true')
    parser.add_argument('--mask-ratio', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    weights = args.weights or make_default_save_path(args)
    batch_size = args.batch_size

    if args.train:
        if args.model == 'simclr':
            stats = train_simclr(device, epochs=args.epochs, batch_size=batch_size or 256, save_path=weights)
        elif args.model == 'dino':
            stats = train_dino(
                device,
                epochs=args.epochs,
                batch_size=batch_size or 64,
                n_local=args.n_local,
                use_centering=not args.no_centering,
                save_path=weights,
            )
        else:
            stats = train_mae(
                device,
                epochs=args.epochs,
                batch_size=batch_size or 128,
                mask_ratio=args.mask_ratio,
                save_path=weights,
            )
        stats_path = weights.replace('.pt', '_stats.json')
        save_stats(stats, stats_path)
        print(f'Saved weights to: {weights}')
        print(f'Saved stats to: {stats_path}')

    if args.evaluate and args.linear:
        if args.model == 'simclr':
            linear_eval_simclr(device, weights=weights, epochs=args.linear_epochs, batch_size=batch_size or 256)
        elif args.model == 'dino':
            linear_eval_dino(device, weights=weights, epochs=args.linear_epochs, batch_size=batch_size or 256)
        else:
            linear_eval_mae(device, weights=weights, epochs=args.linear_epochs,
                            batch_size=batch_size or 256, mask_ratio=args.mask_ratio)

    if args.reconstruct:
        if args.model != 'mae':
            raise ValueError('--reconstruct is only for --model mae')
        visualize_mae_reconstruction(device, weights=weights, mask_ratio=args.mask_ratio)


if __name__ == '__main__':
    main()
