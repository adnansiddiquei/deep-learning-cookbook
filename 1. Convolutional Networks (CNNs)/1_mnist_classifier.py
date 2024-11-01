"""
A simple convolutional network which classifies MNIST 2D images.

 1. Run `tensorboard --logdir=lightning_logs/`, click link and open Tensorboard
 2. Run `python 1_mnist_classifier.py` to start trainig.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT

from dlc.cnn.modules import ConvBlock2d

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

"""
Load the dataset and create the Dataloaders.
"""
train_dataset = MNIST('../data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST('../data', train=False, download=True, transform=ToTensor())

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, drop_last=True, persistent_workers=True)

"""
Define a CNN classifier network, this must classifiy MNIST insput into 1 of 10 classes.
"""
class CNNClassifier(nn.Module):
    """A CNN classifier for the MNIST 2D dataset.
     
    Takes in inputs of shape (B, 1, 28, 28).
    Outputs (B, 10), i.e., classifies a sample into 1 of 10 classes.
    """
    def __init__(self):
        super().__init__()

        self.input_shape = (256, 1, 28, 28)

        self.net = nn.Sequential(
            ConvBlock2d(1, 3, self.input_shape[-2:], kernel_size=3),
            ConvBlock2d(3, 3, self.input_shape[-2:], kernel_size=5),
            ConvBlock2d(3, 1, self.input_shape[-2:], kernel_size=7),
            nn.Flatten(-3),
            nn.Linear(
                1 * 28 * 28,  # after nn.Flatten(-3), shape is (256, 784)
                10  # the final 10 classifications
            )
        )

        out_channels = 1

    def forward(self, x):
        return self.net(x)
    
"""
Define a simple Lightning Trainer class.
"""

class LitCNNClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.example_input_array = torch.Tensor(256, 1, 28, 28)
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def _compute_loss(self, batch: list[torch.Tensor]):
        x, labels = batch
        preds = self(x)
        return F.cross_entropy(preds, labels)
    
    def training_step(self, batch: list[torch.Tensor]) -> STEP_OUTPUT:
        loss = self._compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: list[torch.Tensor]) -> STEP_OUTPUT:
        loss = self._compute_loss(batch)
        self.log('val/loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

"""
Train the model.
"""

if __name__ == '__main__':
    # Create an instance of the model
    cnn_classifier = CNNClassifier()

    # Create an instance of the Lightning module
    lit_cnn_classifier = LitCNNClassifier(model=cnn_classifier)

    # Define the callbacks.
    # Early stopping callback to stop the training when the val/loss doesn't increase for 3 epochs
    early_stopping_callback = EarlyStopping('val/loss', patience=3)

    # Checkpointing callback to save the model with the lowest validation loss
    model_checkpoint_callback = ModelCheckpoint(
        dirpath='./mnist_classifier_checkpoints',
        filename='{epoch:02d}-{step}-min',
        monitor='val/loss',
        save_top_k=1,
    )

    # Start the training
    trainer = L.Trainer(max_epochs=100, callbacks=[early_stopping_callback, model_checkpoint_callback])
    trainer.fit(model=lit_cnn_classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)