import torch
from torch import optim
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import logging
from algseg.utils.dice_score import dice_loss
from algseg.utils.dice_score import multiclass_dice_coeff, dice_coeff

from algseg.cmodels.CModelBase import CModelBase
from algseg.config.CCfgUNet import CCfgUNet
from algseg.models.unet import UNet
from algseg.dataset import get_dataset


class CUNet(CModelBase):
    def __init__(self, config: CCfgUNet):
        super().__init__(config)
        self.config = config
        self.model = UNet(n_channels=config.data_channels,
                          n_classes=config.data_classes,
                          bilinear=config.bilinear)

    def train(self):
        dir_checkpoint = self.config.output_dir
        # 数据集
        # 创建数据集
        data_classes = self.config.data_classes
        data_channels = self.config.data_channels
        data_root = self.config.data_root
        dataset = get_dataset(self.config.data_name, data_root)
        dataset.init_dataset(["train"])
        train_dataloader = dataset.train_dataloader

        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(n_channels=data_channels, n_classes=data_classes, bilinear=False)
        model.to(device=device)

        # train
        learning_rate = 1e-5
        weight_decay = 1e-8
        momentum = 0.999
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay, momentum=momentum)  # , foreach=True
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0

        # 训练
        epochs = 100
        gradient_clipping = 1.0
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=len(dataset.train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_dataloader:
                    images, true_masks = batch[0], batch[1][:, :, :]

                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    with autocast():
                        masks_pred = model(images)
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    # experiment.log({
                    #     'train loss': loss.item(),
                    #     'step': global_step,
                    #     'epoch': epoch
                    # })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    str_loss = "loss:{}".format(loss)
                    self.put_data_list("train_loss", str_loss)
                    self.print_loss(str_loss)

            # Evaluation round
            if epoch % 1 == 0:
                val_score = self.evaluate(model, dataset.train_dataloader, device, False)
                scheduler.step(val_score)
                logging.info('Validation Dice score: {}'.format(val_score))

                # save model
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                # state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    @torch.inference_mode()
    def evaluate(self, net, dataloader, device, amp):
        net.eval()
        num_val_batches = len(dataloader)
        dice_score = 0

        # iterate over the validation set
        # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        with autocast():
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                image, mask_true = batch[0], batch[1][:, :, :]

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                # predict the mask
                mask_pred = net(image)

                if net.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

        net.train()
        return dice_score / max(num_val_batches, 1)

