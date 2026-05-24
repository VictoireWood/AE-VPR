import argparse
import os
import platform
from datetime import datetime


DATASET_CHOICES = [
    "ct01",
    "ct02",
    "qd01",
    "qd02",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the altitude-classification pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=0, help="Random seed.")
    runtime.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device used for model execution.")
    runtime.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")

    training = parser.add_argument_group("training")
    training.add_argument("-ipe", "--iterations_per_epoch", type=int, default=2000, help="Training iterations per epoch.")
    training.add_argument("-bs", "--batch_size", type=int, default=64, help="Training batch size.")
    training.add_argument("--scheduler_patience", type=int, default=10, help="ReduceLROnPlateau patience.")
    training.add_argument("--epochs_num", type=int, default=500, help="Maximum number of training epochs.")
    training.add_argument("--lr", type=float, default=0.0001, help="Backbone and aggregator learning rate.")
    training.add_argument("--classifier_lr", type=float, default=0.01, help="Classifier learning rate.")
    training.add_argument("--resume_train", type=str, default=None, help="Path to a full *_ckpt.pth training checkpoint.")
    training.add_argument("--resume_model", type=str, default=None, help="Path to a *_model.pth model checkpoint.")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset_name", type=str, default="ct01", choices=DATASET_CHOICES, help="Training dataset key.")
    data.add_argument("--test_set_list", nargs="+", type=str, default=None, help="One or more test dataset keys.")
    data.add_argument("--train_set_path", type=str, default=None, help="Path to the training root.")
    data.add_argument("--test_set_path", type=str, default=None, help="Path to the test root.")
    data.add_argument("--train_resize", type=int, nargs=2, default=(336, 448), metavar=("HEIGHT", "WIDTH"), help="Training image resize.")
    data.add_argument("--test_resize", type=int, nargs="+", default=[336], help="Resize argument passed to torchvision.transforms.Resize during evaluation.")
    data.add_argument("--N", type=int, default=1, help="Number of classifier groups.")
    data.add_argument("--M", type=int, default=50, help="Altitude-bin width in meters.")
    data.add_argument("--min_images_per_class", type=int, default=15, help="Minimum images required for a training altitude bin.")
    data.add_argument("--fft", action=argparse.BooleanOptionalAction, default=True, help="Enable Spat2Freq FFT preprocessing.")
    data.add_argument("-flb", "--fft_log_base", type=float, default=1.5, help="Log base used for FFT magnitude compression.")

    model = parser.add_argument_group("model")
    model.add_argument("-bb", "--backbone", type=str, default="resnet50", help="Backbone architecture name.")
    model.add_argument(
        "-agg",
        "--aggregator",
        type=str,
        default="MixVPR",
        choices=["MixVPR", "SALAD", "ConvAP", "CosPlace", "GeMPool", "AvgPool"],
        help="Feature aggregation head.",
    )
    model.add_argument("-ntb", "--num_trainable_blocks", type=int, default=2, help="Number of trainable DINOv2 blocks.")
    model.add_argument("-ltf", "--layers_to_freeze", type=int, default=5, help="Number of CNN stages to freeze.")
    model.add_argument("-ltc", "--layers_to_crop", type=int, nargs="+", default=[4], choices=[3, 4], help="ResNet stages to remove from the tail.")
    model.add_argument("-moc", "--mixvpr_out_channels", type=int, default=None, help="Override MixVPR output channels.")
    model.add_argument(
        "--classifier",
        type=str,
        default="QAMC",
        choices=["QAMC", "LMCC", "AAMC", "LinearLayer"],
        help="Altitude-classification head.",
    )
    model.add_argument("--aamc_m", type=float, default=0.2, help="Margin parameter used by LMCC/AAMC.")
    model.add_argument("--aamc_s", type=float, default=100.0, help="Scale parameter used by LMCC/AAMC.")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--threshold", type=int, default=None, help="Recall threshold in meters; when omitted the code uses its dataset default.")

    output = parser.add_argument_group("output")
    output.add_argument("--exp_name", type=str, default="default", help="Experiment name used under logs/.")

    args = parser.parse_args()

    if args.exp_name == "default":
        args.exp_name = f"hc-{args.backbone}-{args.aggregator}"

    args.save_dir = os.path.join("logs", args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if args.device == "cpu" or platform.system().lower() == "windows":
        args.num_workers = 0

    return args
