"""Inference Entrypoint script."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ActionConfigFile, Namespace
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyModule, get_model


def get_parser() -> LightningArgumentParser:
    """Get parser.

    Returns:
        LightningArgumentParser: The parser object.
    """
    ckpt_path = r"D:\00sheepy\00MyMLStudy\ml10Repositorys\anomalib-1.1.1\src\results\Padim\MVTec\bottle\v8\weights\lightning/model.ckpt"
    parser = LightningArgumentParser(description="Inference on Anomaly models in Lightning format.")
    # parser.add_lightning_class_args(AnomalyModule, "model", default="Padim", subclass_mode=True)
    parser.add_lightning_class_args(Callback, "--callbacks", subclass_mode=True, required=False)
    parser.add_argument("--ckpt_path", type=str, default=ckpt_path,required=False, help="Path to model weights")
    # parser.add_class_arguments(PredictDataset, "--data", default=data_path, instantiate=False)
    parser.add_argument("--output", type=str, default="./output", required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )
    # parser.add_argument(
    #     "-c",
    #     "--config",
    #     action=ActionConfigFile,
    #     default="",
    #     help="Path to a configuration file in json or yaml format.",
    # )

    return parser


def infer(args: Namespace) -> None:
    """Run inference."""
    callbacks = None if not hasattr(args, "callbacks") else args.callbacks
    engine = Engine(
        default_root_dir=args.output,
        callbacks=callbacks,
        devices=1,
    )
    model = get_model("Padim", pre_trained=False)
    # create the dataset
    dataset = PredictDataset(r"D:\04Bin\img_test")
    dataloader = DataLoader(dataset)

    engine.predict(model=model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    infer(args)
