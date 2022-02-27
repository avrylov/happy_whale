
import torch
from torch.nn import functional as F
import torchmetrics.functional as M
import pytorch_lightning as pl

from .model import SNUNet


class TorchLightNet(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.model = SNUNet()

    def forward(self, inputs):
        im_a = inputs['first']
        im_b = inputs['second']
        out = self.model(im_a, im_b)
        return out

    def training_step(self, batch, batch_idx):
        idx, inputs, labels = batch
        im_a = inputs['first']
        im_b = inputs['second']
        labels = torch.unsqueeze(labels, dim=1)

        preds = self.model(im_a.float(), im_b.float())
        logits = torch.sigmoid(preds)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        idx, inputs, labels = batch
        im_a = inputs['first']
        im_b = inputs['second']
        labels = torch.unsqueeze(labels, dim=1)

        preds = self.model(im_a.float(), im_b.float())
        logits = torch.sigmoid(preds)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())

        threshold = torch.tensor([0.5])
        device = torch.device('cuda:0')
        threshold = threshold.to(device, dtype=torch.float)
        bi_preds = (logits > threshold).float() * 1
        acc = M.accuracy(bi_preds.int(), labels.int())

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=5e-1,
                patience=2,
                verbose=True
            ),
            "monitor": "val_loss"
        }
        d = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
        return d

