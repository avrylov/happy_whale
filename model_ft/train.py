from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import early_stopping, model_checkpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

from trainer import TorchLightNet
from generator import MyDataset
from make_data_frame import make_data_frame
from settings import csv_folder, logger_path, check_pointer_path
from utils import make_yaml


dataset, df = make_data_frame(csv_folder,
                              'train_bbox_df.csv',
                              'validate_bbox_df.csv',
                              'test_bbox_df.csv')
train = df.get('train')
validate = df.get('validate')

batch_size = 16
train_full_batch = (train.shape[0] // batch_size) * batch_size

train_transform = A.Compose(
    [
        A.Resize(width=128, height=128),
        # A.CenterCrop(width=128, height=128),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.Normalize(mean=0, std=1),
        ToTensorV2(),
    ],
    additional_targets={'image_b': 'image'}
)
train_dataset = MyDataset(train[:train_full_batch], crop=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=6)

val_transform = A.Compose(
    [
        A.Resize(width=128, height=128),
        A.Normalize(mean=0, std=1),
        ToTensorV2(),
    ],
    additional_targets={'image_b': 'image'}
)

val_dataset = MyDataset(validate, crop=True, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=6)

logger = loggers.TensorBoardLogger(logger_path)
es = early_stopping.EarlyStopping(monitor='val_loss',
                                  mode='min',
                                  patience=75)

check_pointer = model_checkpoint.ModelCheckpoint(dirpath=check_pointer_path,
                                                 filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}',
                                                 save_top_k=1,
                                                 save_on_train_epoch_end=False,
                                                 monitor='val_loss',
                                                 mode='min')

pl_model = TorchLightNet(lr=1e-3, weight_decay=5e-4)
make_yaml(pl_model, dataset, check_pointer_path)

trainer = pl.Trainer(
    devices='1',
    accelerator='gpu',
    logger=logger,
    callbacks=[es, check_pointer],
    max_epochs=250)

trainer.fit(pl_model,
            train_loader, val_loader)