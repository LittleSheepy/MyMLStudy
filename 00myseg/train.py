import torch
from torch import optim
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
from tqdm import tqdm
from dataset.CCityScapes import CCityScapes
from models.unet import UNet
from utils.dice_score import dice_loss
from pathlib import Path
import logging

from evaluate import evaluate

if __name__ == '__main__':
    data_root = r"F:\sheepy\00MyMLStudy\ml10Repositorys\05open-mmlab\mmsegmentation-1.2.1_my\data\cityscapes/"
    dir_checkpoint = Path('./checkpoints/')
    # 创建数据集
    classes = 100
    dataset = CCityScapes(data_root)
    dataset.init_dataset(["train"])
    train_dataloader = dataset.train_dataloader

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=classes, bilinear=False)
    model.to(device=device)

    # train
    learning_rate = 1e-5
    weight_decay = 1e-8
    momentum = 0.999
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)# , foreach=True
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
            for batch in dataset.train_dataloader:
                images, true_masks = batch[0], batch[1][:,0,:,:]

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

        # Evaluation round
        if epoch % 1 == 0:
            val_score = evaluate(model, dataset.train_dataloader, device, False)
            scheduler.step(val_score)
            logging.info('Validation Dice score: {}'.format(val_score))

            # save model
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
