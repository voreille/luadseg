from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from your_module import PatSegLightningModule, SegDataModule
from your_package.models.encoder_wrapper import UNI2Encoder
from your_package.models.patseg import PatSeg


def main():
    # Provide your data here
    image_paths = [...]  # List of paths to 768x768 images
    mask_paths = [...]  # List of corresponding mask paths

    # Init model
    encoder = UNI2Encoder()
    model = PatSeg(encoder=encoder, extract_layers=(6, 12, 18, 24), num_classes=7)
    lightning_model = PatSegLightningModule(model)

    # Logger and checkpoint
    logger = TensorBoardLogger("logs", name="patseg")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", save_top_k=1, mode="min", filename="patseg-{epoch:02d}-{val_loss:.2f}"
    )

    # Data module
    data_module = SegDataModule(image_paths, mask_paths, batch_size=2)

    # Trainer
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16,
        devices=1,
        accelerator="gpu",
    )

    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
