import argparse

from data_utils.celeba_data import get_celeba_loaders
from data_utils.mnist_data import get_mnist_loader
from trainers.train_cyclegan import train_cyclegan
from trainers.train_ddpm import train_ddpm
from trainers.train_gan import train_gan
from utils.common import create_directories, get_device, set_seed
from trainers.train_mnist_classifier import train_mnist_classifier
from evaluation.mode_collapse import evaluate_mode_distribution
from inference.generate_ddpm import generate_from_checkpoint

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train GAN, CycleGAN, or DDPM"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["gan", "cyclegan", "ddpm", "classifier"],
    )

    parser.add_argument(
        "--dataset",
        choices=["mnist", "celeba"],
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--train",
        action="store_true",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--generate",
        action="store_true",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--schedule",
        choices=["linear", "cosine"],
        default="linear",
    )

    parser.add_argument(
        "--discriminator-lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lambda-cyc",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--lambda-idt",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
    "--evaluate-modes",
    action="store_true",
    )
    
    parser.add_argument(
    "--output-name",
    type=str,
    default="gan",
    )
    return parser.parse_args()


def run_gan(args, device):
    batch_size = args.batch_size or 128

    loader = get_mnist_loader(
        root=args.data_dir,
        batch_size=batch_size,
    )

    checkpoint_name = (
        "saved/gan_mnist.pt"
        if args.discriminator_lr == 2e-4
        else "saved/gan_high_d_lr.pt"
    )

    train_gan(
        train_loader=loader,
        device=device,
        epochs=args.epochs,
        discriminator_lr=args.discriminator_lr,
        checkpoint_path=checkpoint_name,
    )
    
def run_gan(args, device):
    if args.evaluate_modes:
        if args.weights is None:
            raise ValueError(
                "--weights is required when using "
                "--evaluate-modes"
            )

        evaluate_mode_distribution(
            gan_checkpoint_path=args.weights,
            classifier_checkpoint_path=(
                "saved/mnist_classifier.pt"
            ),
            device=device,
            output_name=args.output_name,
            number_of_samples=args.n,
        )

        return

    batch_size = args.batch_size or 128

    loader = get_mnist_loader(
        root=args.data_dir,
        batch_size=batch_size,
    )

    checkpoint_name = (
        "saved/gan_mnist.pt"
        if args.discriminator_lr == 2e-4
        else "saved/gan_high_d_lr.pt"
    )

    train_gan(
        train_loader=loader,
        device=device,
        epochs=args.epochs,
        discriminator_lr=args.discriminator_lr,
        checkpoint_path=checkpoint_name,
    )


def run_cyclegan(args, device):
    batch_size = args.batch_size or 16

    dark_loader, blonde_loader = (
        get_celeba_loaders(
            root=args.data_dir,
            batch_size=batch_size,
        )
    )

    checkpoint_name = (
        "saved/cyclegan_celeba.pt"
        if args.lambda_cyc == 10.0
        else f"saved/cyclegan_lambda"
             f"{args.lambda_cyc:g}.pt"
    )

    train_cyclegan(
        dark_loader=dark_loader,
        blonde_loader=blonde_loader,
        device=device,
        epochs=args.epochs,
        lambda_cycle=args.lambda_cyc,
        lambda_identity=args.lambda_idt,
        checkpoint_path=checkpoint_name,
    )


def run_ddpm(args, device):
    if args.generate:
        if args.weights is None:
            raise ValueError(
                "--weights is required with --generate"
            )

        output_name = args.output_name or "ddpm_samples"

        generate_from_checkpoint(
            checkpoint_path=args.weights,
            output_path=(
                f"outputs/ddpm/{output_name}.png"
            ),
            device=device,
            number_of_images=args.n,
        )

        return

    batch_size = args.batch_size or 128

    loader = get_mnist_loader(
        root=args.data_dir,
        batch_size=batch_size,
    )

    checkpoint_name = (
        f"saved/ddpm_mnist_{args.schedule}.pt"
    )

    train_ddpm(
        train_loader=loader,
        device=device,
        epochs=args.epochs,
        schedule_name=args.schedule,
        checkpoint_path=checkpoint_name,
    )

def run_classifier(args, device):
    batch_size = args.batch_size or 128

    loader = get_mnist_loader(
        root=args.data_dir,
        batch_size=batch_size,
    )

    train_mnist_classifier(
        train_loader=loader,
        device=device,
        epochs=args.epochs,
        checkpoint_path="saved/mnist_classifier.pt",
    )

def main():
    args = parse_arguments()

    set_seed(args.seed)
    create_directories()

    device = get_device()

    print(f"Using device: {device}")
    print(f"Selected model: {args.model}")

    if (
      not args.train
      and not args.evaluate_modes
      and not args.generate
    ):
      raise ValueError(
        "Choose --train, --evaluate-modes, or --generate."
    )

    if args.model == "gan":
        run_gan(args, device)

    elif args.model == "cyclegan":
        run_cyclegan(args, device)

    elif args.model == "ddpm":
        run_ddpm(args, device)
    
    elif args.model == "classifier":
        run_classifier(args, device)
    
   
    

if __name__ == "__main__":
    main()
