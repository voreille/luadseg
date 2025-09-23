from pathlib import Path

import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from luadseg.data.pl_data import SegDataModule
from luadseg.models.pl_wrapper import PatSegLightningModule
from luadseg.models.segmentation.encoder_wrapper import UNI2Encoder
from luadseg.models.segmentation.patseg import PatSeg

project_dir = Path(__file__).parents[2].resolve()


@click.command()
@click.option("--root-data-dir", default="data/processed/ANORAK_not_resized", help="Root directory for data.")
@click.option("--tensorboard-dir", default="logs/tb_logs", help="Directory for TensorBoard logs.")
@click.option("--checkpoint-dir", default="checkpoints", help="Directory for model checkpoints.")
@click.option("--batch-size", default=4, type=int, help="Batch size for training.")
@click.option("--num-workers", default=4, type=int, help="Number of workers for data loading.")
def main(root_data_dir, tensorboard_dir, checkpoint_dir, batch_size, num_workers):
    root_data_dir = project_dir / root_data_dir
    tensorboard_dir = project_dir / tensorboard_dir
    checkpoint_dir = project_dir / checkpoint_dir

    # Init model
    encoder = UNI2Encoder()
    model = PatSeg(encoder=encoder, num_classes=7)
    lightning_model = PatSegLightningModule(model)

    # Logger and checkpoint
    logger = TensorBoardLogger(tensorboard_dir, name="patseg")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="patseg-{epoch:02d}-{val_loss:.2f}",
        dirpath=checkpoint_dir,
    )

    # Data module
    data_module = SegDataModule(
        batch_size=batch_size, num_workers=num_workers, root_dir=root_data_dir
    )

    # Trainer
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16,
        devices=1,
        accelerator="gpu",
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
